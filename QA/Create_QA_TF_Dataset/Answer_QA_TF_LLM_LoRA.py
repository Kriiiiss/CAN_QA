import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

QUESTION_FILES = {
    "DoS": Path("DoS_tf_qa/questions/dos_questions.json"),
    "Fuzzy": Path("Fuzzy_tf_qa/questions/fuzzy_questions.json"),
    "Gear": Path("Gear_tf_qa/questions/gear_questions.json"),
    "RPM": Path("RPM_tf_qa/questions/rpm_questions.json"),
}

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_prompt_text(tokenizer, context: str, question: str, context_type: str = "full") -> str:
    extra_instruction = ""
    if context_type == "hidden_random_flag":
        extra_instruction = "One frame's flag is hidden at a random position; use surrounding evidence in the window to infer it.\n"

    system_prompt = (
        "You are a CAN bus intrusion-detection analyst. Study timestamp ordering, ID frequency, "
        "payload stability, byte ranges, and gaps between frames. Use those characteristics plus the "
        "statement to reach a True/False conclusion. Respond with True or False only."
    )
    user_prompt = (
        "Below is a CAN bus time window. Review the sequence carefully, note anomalies or missing IDs, "
        "and reason about the claim.\n"
        f"{context}\n\n{extra_instruction}"
        f"Statement: {question}\nAnswer True or False only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def normalize_tf_answer(text: str) -> str:
    if not text:
        return "False"
    first = text.strip().split()[0].replace(".", "").lower()
    if first.startswith("t"):
        return "True"
    if first.startswith("f"):
        return "False"
    return "False"


def load_questions(path: Path) -> List[dict]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(records: List[dict], test_size: float, seed: int) -> Tuple[List[dict], List[dict]]:
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_label[str(rec.get("ground_truth", "False"))].append(rec)

    rng = random.Random(seed)
    train, test = [], []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        n_total = len(label_records)
        n_test = max(1, int(round(n_total * test_size))) if n_total > 1 else 0
        test.extend(label_records[:n_test])
        train.extend(label_records[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def resolve_torch_dtype(arg_dtype: str, use_cuda: bool):
    if not use_cuda:
        if arg_dtype in {"fp16", "bf16"}:
            print(f"[WARN] --torch_dtype {arg_dtype} requested without CUDA; using fp32 on CPU.")
        return torch.float32

    if arg_dtype == "fp16":
        return torch.float16
    if arg_dtype == "bf16":
        return torch.bfloat16
    if arg_dtype == "fp32":
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def estimate_model_billions(model_id: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def query_llm(model, tokenizer, context: str, question: str, context_type: str = "full", max_new_tokens: int = 16) -> str:
    prompt_text = build_prompt_text(tokenizer, context, question, context_type=context_type)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    return normalize_tf_answer(completion)


class TFSFTDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_length: int):
        self.examples = []
        eos = tokenizer.eos_token or ""

        for rec in records:
            prompt = build_prompt_text(
                tokenizer,
                rec["context"],
                rec["question"],
                rec.get("context_type", "full"),
            )
            answer = rec.get("ground_truth", "False").strip()
            full_text = f"{prompt}{answer}{eos}"

            full_enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            labels = input_ids.copy()

            p_len = min(len(prompt_enc["input_ids"]), len(labels))
            labels[:p_len] = [-100] * p_len

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate_tf_accuracy(model, tokenizer, records: List[dict], max_eval_samples: int) -> float:
    if not records:
        return 0.0

    subset = records[:max_eval_samples] if max_eval_samples > 0 else records
    correct = 0

    for rec in tqdm(subset, desc="Eval", leave=False):
        pred = query_llm(model, tokenizer, rec["context"], rec["question"], rec.get("context_type", "full"))
        if pred == rec.get("ground_truth"):
            correct += 1

    return correct / max(1, len(subset))


def parse_args():
    parser = argparse.ArgumentParser(description="Notebook-friendly Hugging Face QLoRA for CAN TF QA.")
    parser.add_argument("--model_id", default="mistralai/Ministral-3-8B-Instruct-2512")
    parser.add_argument("--selected_datasets", nargs="+", default=["DoS", "Fuzzy", "Gear", "RPM"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_output_dir", default="lora_tf_hf_artifacts")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--answer_split", choices=["test", "train", "all"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda and os.name == "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dtype = resolve_torch_dtype(args.torch_dtype, use_cuda)
    est_b = estimate_model_billions(args.model_id)
    if not use_cuda and est_b is not None and est_b >= 7.0:
        raise RuntimeError("CPU-only training is not supported for 7B+ in this script. Use a CUDA runtime.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    all_by_dataset = {}
    all_records = []

    for ds_name in args.selected_datasets:
        q_path = QUESTION_FILES[ds_name]
        questions = load_questions(q_path)
        if not questions:
            print(f"[WARN] No questions found for {ds_name} at {q_path}")
            continue
        tagged = []
        for rec in questions:
            tagged_rec = dict(rec)
            tagged_rec["_dataset_key"] = ds_name
            tagged_rec["_row_id"] = len(all_records)
            tagged.append(tagged_rec)
            all_records.append(tagged_rec)
        all_by_dataset[ds_name] = tagged
        print(f"[INFO] {ds_name}: loaded {len(tagged)} questions")

    if not all_records:
        raise RuntimeError("No question records loaded.")

    train_records, test_records = stratified_split(all_records, args.test_size, args.seed)
    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
    print(f"[INFO] split -> train: {len(train_records)}, test: {len(test_records)}")

    quant_cfg = None
    if args.use_4bit:
        if not use_cuda:
            raise RuntimeError("--use_4bit requires CUDA.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "device_map": "auto" if use_cuda else None,
        "torch_dtype": dtype,
        "quantization_config": quant_cfg,
        "low_cpu_mem_usage": True,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(
        f"[INFO] runtime -> python {sys.version.split()[0]}, cuda: {use_cuda}, "
        f"dtype: {dtype}, attn: {args.attn_implementation}, qlora_4bit: {args.use_4bit}"
    )
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    before_acc = evaluate_tf_accuracy(base_model, tokenizer, test_records, args.max_eval_samples)
    print(f"[INFO] pre-finetune test accuracy: {before_acc:.4f}")

    if args.use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = TFSFTDataset(train_records, tokenizer, args.max_length)
    collator = CausalDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.train_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        fp16=use_cuda and dtype == torch.float16,
        bf16=use_cuda and dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    adapter_dir = Path(args.train_output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    tuned_model.eval()

    after_acc = evaluate_tf_accuracy(tuned_model, tokenizer, test_records, args.max_eval_samples)
    print(f"[INFO] post-finetune test accuracy: {after_acc:.4f}")

    train_row_ids = {r["_row_id"] for r in train_records}
    test_row_ids = {r["_row_id"] for r in test_records}
    inference_model_id = f"{args.model_id}+lora"

    for ds_name, records in all_by_dataset.items():
        q_path = QUESTION_FILES[ds_name]
        out_dir = q_path.parent.parent / "llm_answers"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_tag = args.model_id.split("/")[-1].replace(".", "_").replace("-", "_") + "_lora"
        ans_path = out_dir / f"{ds_name.lower()}_answers_{model_tag}_{args.answer_split}.jsonl"

        if args.answer_split == "test":
            questions = [r for r in records if r["_row_id"] in test_row_ids]
        elif args.answer_split == "train":
            questions = [r for r in records if r["_row_id"] in train_row_ids]
        else:
            questions = list(records)

        ans_path.write_text("", encoding="utf-8")
        print(f"[INFO] {ds_name}: answering {len(questions)} records from split='{args.answer_split}'")
        with ans_path.open("a", encoding="utf-8") as f:
            for rec in tqdm(questions, desc=f"{ds_name} answering"):
                pred = query_llm(
                    tuned_model,
                    tokenizer,
                    rec["context"],
                    rec["question"],
                    rec.get("context_type", "full"),
                )

                answer_rec = {
                    "qa_id": rec["qa_id"],
                    "dataset": rec["metadata"]["dataset"],
                    "model": inference_model_id,
                    "llm_answer": pred,
                    "ground_truth": rec["ground_truth"],
                    "is_correct": pred == rec["ground_truth"],
                    "answer_valid": bool(pred),
                }
                f.write(json.dumps(answer_rec, ensure_ascii=False) + "\n")

        print(f"[INFO] {ds_name}: answers saved to {ans_path}")


if __name__ == "__main__":
    main()
