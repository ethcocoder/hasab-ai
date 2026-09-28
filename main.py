"""
main.py — Hasab Qwen2.5-0.5B pipeline.

Usage:
  python main.py --mode full       # acquire, clean, LoRA train, evaluate
  python main.py --mode acquire
  python main.py --mode clean
  python main.py --mode finetune
  python main.py --mode test
  python main.py --mode chat
"""

import argparse
import json
from pathlib import Path

import torch

from config import CFG
from soul.culture_injector import CultureInjector
from soul.safety_gate import SafetyGate
from heart.persona import PersonaTracker
from pipeline.acquire import DataAcquirer
from pipeline.clean import AmharicCleaner
from pipeline.finetune import QwenTrainer


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
    cleaner = AmharicCleaner(
        min_length=CFG.data.min_sentence_length,
        max_length=CFG.data.max_sentence_length,
        min_amharic_ratio=CFG.data.min_amharic_ratio,
    )
    raw_files = list(CFG.paths.data_raw.glob("*.txt"))
    if not raw_files:
        print("⚠️  No raw files found. Running acquire first.")
        stage_acquire()
        raw_files = list(CFG.paths.data_raw.glob("*.txt"))

    all_sentences = []
    for raw_file in raw_files:
        out_path = CFG.paths.data_clean / raw_file.name
        cleaner.process_file(raw_file, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            all_sentences.extend(s for s in f.read().splitlines() if s.strip())

    clean_sentences, removed = SafetyGate().filter_dataset(all_sentences)
    print(f"🛡️  Safety filter removed {removed} sentences")
    final_sentences = CultureInjector(
        upweight_factor=CFG.soul.amharic_culture_weight
    ).inject_into_dataset(clean_sentences)
    final_path = CFG.paths.data_clean / "corpus_final.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_sentences))
    print(f"✅ Final corpus: {len(final_sentences)} sentences → {final_path}")
    return final_sentences


def load_corpus():
    path = CFG.paths.data_clean / "corpus_final.txt"
    if not path.exists():
        return stage_clean()
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stage_finetune():
    print("\n" + "=" * 60)
    print("③ QWEN2.5-0.5B LoRA FINE-TUNING")
    print("=" * 60)
    sentences = load_corpus()
    n = len(sentences)
    train_end = max(1, int(n * CFG.data.train_split))
    val_end = max(train_end + 1, int(n * (CFG.data.train_split + CFG.data.val_split)))
    trainer = QwenTrainer(CFG)
    model, tokenizer = trainer.train(sentences[:train_end], sentences[train_end:val_end])
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
        choices=["full", "acquire", "clean", "finetune", "test", "chat"],
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
        "finetune": stage_finetune,
        "test": stage_test,
        "chat": stage_chat,
    }
    stages[args.mode]()


if __name__ == "__main__":
    main()
