"""
main.py — Hasab Qwen2.5-0.5B pipeline.

Usage:
  python main.py --mode full       # acquire, clean, LoRA train, evaluate
  python main.py --mode acquire
  python main.py --mode clean
  python main.py --mode prepare-data
  python main.py --mode finetune
  python main.py --mode test
  python main.py --mode chat
"""

import argparse
import json
import os
from pathlib import Path

# Qwen training is PyTorch-only. Prevent Transformers from importing Colab's
# TensorFlow stack, which can cause binary conflicts and process crashes.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch

from config import CFG
from soul.culture_injector import CultureInjector
from soul.safety_gate import SafetyGate
from heart.persona import PersonaTracker
from pipeline.acquire import DataAcquirer
from pipeline.finetune import QwenTrainer
from pipeline.prepare_data import AmharicDataBuilder


def stage_acquire():
    print("\n" + "=" * 60)
    print("① ACQUIRING DATA")
    print("=" * 60)
    CFG.make_dirs()
    acquirer = DataAcquirer(CFG.paths.data_raw, max_mb=CFG.data.max_raw_mb)
    wiki_path = acquirer.acquire_wikipedia()
    examples = CultureInjector(upweight_factor=3).build_chat_examples()
    chat_path = CFG.paths.data_raw / "cultural_chat.jsonl"
    with open(chat_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"✅ Cultural examples: {len(examples)} → {chat_path}")
    return wiki_path


def stage_clean():
    print("\n" + "=" * 60)
    print("② CLEANING DATA")
    print("=" * 60)
    raw_files = list(CFG.paths.data_raw.glob("*.txt"))
    if not raw_files:
        print("⚠️  No raw files found. Running acquire first.")
        stage_acquire()
        raw_files = list(CFG.paths.data_raw.glob("*.txt"))

    outputs = AmharicDataBuilder(CFG).build()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    stats = manifest["stats"]
    print(f"✅ Knowledge records: {stats['kept_knowledge']} → {outputs['knowledge']}")
    print(f"✅ Chat records: {stats['kept_chat']} → {outputs['chat']}")
    print(f"🧹 Removed duplicates: {stats['removed_duplicate']}")
    print(f"⚠️  Held out for human review: {stats['held_for_review']} → {outputs['review']}")
    return outputs


def load_records():
    knowledge_path = CFG.paths.data_processed / "knowledge.jsonl"
    chat_path = CFG.paths.data_processed / "chat.jsonl"
    if not knowledge_path.exists() or not chat_path.exists():
        stage_clean()

    def read_jsonl(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    knowledge = read_jsonl(knowledge_path)
    chat = read_jsonl(chat_path)
    if not knowledge and not chat:
        raise ValueError("The prepared dataset is empty; inspect data/processed/manifest.json")
    return knowledge, chat


def stage_prepare_data():
    """Build processed knowledge/chat JSONL files and a quality manifest."""
    print("\n" + "=" * 60)
    print("② DATA QUALITY PREPARATION")
    print("=" * 60)
    stage_clean()


def stage_finetune():
    print("\n" + "=" * 60)
    print("③ QWEN2.5-0.5B LoRA FINE-TUNING")
    print("=" * 60)
    knowledge, chat = load_records()
    def split(items):
        n = len(items)
        train_end = max(1, int(n * CFG.data.train_split)) if n else 0
        val_end = max(train_end + 1, int(n * (CFG.data.train_split + CFG.data.val_split))) if n else 0
        return items[:train_end], items[train_end:val_end]
    knowledge_train, knowledge_val = split(knowledge)
    chat_train, chat_val = split(chat)
    # Repeat only within the training partition to avoid validation leakage.
    target_chat = max(1, round(len(knowledge_train) * CFG.data.chat_mix / max(CFG.data.knowledge_mix, 0.01))) if chat_train else 0
    mixed_chat_train = (chat_train * ((target_chat + len(chat_train) - 1) // len(chat_train)))[:target_chat] if chat_train else []
    trainer = QwenTrainer(CFG)
    model, tokenizer = trainer.train(
        knowledge_train + mixed_chat_train,
        knowledge_val + chat_val,
    )
    return model, tokenizer


def stage_test():
    print("\n" + "=" * 60)
    print("④ QWEN SMOKE EVALUATION")
    print("=" * 60)
    if not (CFG.paths.qwen_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No Qwen adapter found at {CFG.paths.qwen_adapter}. "
            "Run --mode finetune first."
        )
    trainer = QwenTrainer(CFG)
    for prompt in CFG.eval.generation_prompts:
        response = trainer.generate(prompt, max_new_tokens=40)
        print(f"  ↳ {prompt!r}\n    {response!r}\n")
    print(f"✅ Adapter is ready: {CFG.paths.qwen_adapter}")


def stage_chat():
    if not (CFG.paths.qwen_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            "No Qwen adapter found. Run: python main.py --mode finetune"
        )
    trainer = QwenTrainer(CFG)
    PersonaTracker()
    print("\nHasab Qwen chat is ready. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nደህና ሁን!")
            break
        if message.lower() in {"quit", "exit", "ደህና ሁን"}:
            print("ደህና ሁን!")
            break
        if not message:
            continue
        safe, _, refusal = SafetyGate().check_input(message)
        if not safe:
            print(f"Hasab: {refusal}\n")
            continue
        response = trainer.generate(message, max_new_tokens=CFG.body.max_new_tokens)
        print(f"Hasab: {response}\n")


def stage_full():
    print("\n🇪🇹 Hasab — Qwen2.5-0.5B LoRA Pipeline")
    print("Soul → Heart → Qwen Mind")
    CFG.make_dirs()
    stage_clean()
    stage_finetune()
    stage_test()
    print("✅ Qwen smoke pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="Hasab Qwen2.5-0.5B pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "acquire", "clean", "prepare-data", "finetune", "test", "chat"],
        default="full",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.max_steps is not None:
        CFG.training.max_steps = args.max_steps
    if args.batch_size is not None:
        CFG.training.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        CFG.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_workers is not None:
        CFG.training.num_workers = args.num_workers
    if args.max_length is not None:
        CFG.model.max_seq_len = args.max_length
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("--device cuda was requested, but CUDA is not available")
        CFG.training.device = args.device

    CFG.make_dirs()
    stages = {
        "full": stage_full,
        "acquire": stage_acquire,
        "clean": stage_clean,
        "prepare-data": stage_prepare_data,
        "finetune": stage_finetune,
        "test": stage_test,
        "chat": stage_chat,
    }
    stages[args.mode]()


if __name__ == "__main__":
    main()
