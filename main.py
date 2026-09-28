"""
main.py — Amharic Mobile-AGI Pipeline Orchestrator
════════════════════════════════════════════════════════
Entry point for all pipeline stages.

Usage:
  python main.py --mode full           # Run everything
  python main.py --mode acquire        # Download data only
  python main.py --mode clean          # Clean data
  python main.py --mode tokenizer      # Train tokenizer
  python main.py --mode finetune       # Fine-tune model
  python main.py --mode compress       # Quantize + prune
  python main.py --mode export         # Export to ONNX/TFLite
  python main.py --mode test           # Evaluate
  python main.py --mode chat           # Interactive chat
════════════════════════════════════════════════════════
"""

import argparse
import json
import torch
from pathlib import Path

from config import CFG
from soul.values import CoreValues
from soul.culture_injector import CultureInjector
from soul.safety_gate import SafetyGate
from heart.tone import ToneModulator
from heart.persona import PersonaTracker
from body.tokenizer import AmharicTokenizer
from body.quantize import ModelQuantizer
from body.export import MobileExporter
from body.runtime import MobileRuntime
from mind.model import AmharicGPT2
from pipeline.acquire import DataAcquirer
from pipeline.clean import AmharicCleaner
from pipeline.dataset import AmharicDataset
from pipeline.finetune import Trainer
from pipeline.evaluate import Evaluator


# ─────────────────────────────────────────────────────────────────────────────
# STAGE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def stage_acquire():
    print("\n" + "="*60)
    print("① ACQUIRING DATA")
    print("="*60)
    CFG.make_dirs()
    acquirer = DataAcquirer(CFG.paths.data_raw, max_mb=CFG.data.max_raw_mb)

    # Wikipedia
    wiki_path = acquirer.acquire_wikipedia()

    # Cultural injection (Soul module)
    injector   = CultureInjector(upweight_factor=3)
    chat_exs   = injector.build_chat_examples()
    chat_path  = CFG.paths.data_raw / "cultural_chat.jsonl"
    with open(chat_path, "w", encoding="utf-8") as f:
        for ex in chat_exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Cultural examples: {len(chat_exs)} → {chat_path}")

    return wiki_path


def stage_clean():
    print("\n" + "="*60)
    print("② CLEANING DATA")
    print("="*60)
    cleaner = AmharicCleaner(
        min_length         = CFG.data.min_sentence_length,
        max_length         = CFG.data.max_sentence_length,
        min_amharic_ratio  = CFG.data.min_amharic_ratio,
    )
    gate = SafetyGate()

    raw_files = list(CFG.paths.data_raw.glob("*.txt"))
    if not raw_files:
        print("⚠️  No raw files found. Running acquire first.")
        stage_acquire()
        raw_files = list(CFG.paths.data_raw.glob("*.txt"))

    all_sentences = []
    for raw_file in raw_files:
        out_path = CFG.paths.data_clean / raw_file.name
        stats    = cleaner.process_file(raw_file, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            all_sentences.extend(f.read().split("\n"))

    # Safety filter
    clean_sentences, n_removed = gate.filter_dataset(all_sentences)
    print(f"🛡️  Safety filter removed {n_removed} sentences")

    # Cultural upweighting (Soul module)
    injector         = CultureInjector(upweight_factor=CFG.soul.amharic_culture_weight)
    final_sentences  = injector.inject_into_dataset(clean_sentences)

    final_path = CFG.paths.data_clean / "corpus_final.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_sentences))
    print(f"✅ Final corpus: {len(final_sentences)} sentences → {final_path}")
    return final_sentences


def stage_tokenizer(texts=None):
    print("\n" + "="*60)
    print("③ TRAINING TOKENIZER (Body)")
    print("="*60)

    if texts is None:
        corpus_path = CFG.paths.data_clean / "corpus_final.txt"
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                texts = f.read().split("\n")
        else:
            print("⚠️  No clean corpus. Running clean first.")
            texts = stage_clean()

    tokenizer = AmharicTokenizer(vocab_size=CFG.tokenizer.vocab_size)
    tokenizer.train(texts, show_progress=True)
    tokenizer.save(CFG.paths.root / "models" / "tokenizer")

    print(f"✅ Tokenizer: vocab_size={tokenizer.vocab_size_actual()}")
    print(f"   Test encode: {tokenizer.encode('ሰላም ዓለም')}")
    return tokenizer


def stage_finetune(tokenizer=None):
    print("\n" + "="*60)
    print("④ FINE-TUNING (Mind + LCE)")
    print("="*60)

    # Load tokenizer
    if tokenizer is None:
        tokenizer = AmharicTokenizer()
        tokenizer.load(CFG.paths.root / "models" / "tokenizer")

    # Load corpus
    corpus_path = CFG.paths.data_clean / "corpus_final.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        sentences = [s for s in f.read().split("\n") if s.strip()]

    # Split
    n        = len(sentences)
    n_train  = int(n * CFG.data.train_split)
    n_val    = int(n * CFG.data.val_split)
    train_s  = sentences[:n_train]
    val_s    = sentences[n_train:n_train + n_val]

    train_ds = AmharicDataset(train_s, tokenizer, CFG.mind.max_seq_len)
    val_ds   = AmharicDataset(val_s,   tokenizer, CFG.mind.max_seq_len)

    print(f"   Train: {len(train_ds)} examples | Val: {len(val_ds)} examples")

    # Build model
    model = AmharicGPT2(
        vocab_size         = tokenizer.vocab_size_actual(),
        d_model            = CFG.mind.d_model,
        n_layers           = CFG.mind.n_layers,
        n_heads            = CFG.mind.n_heads,
        max_seq_len        = CFG.mind.max_seq_len,
        lce_enabled        = CFG.mind.lce_enabled,
        lce_latent_dim     = CFG.mind.lce_latent_dim,
        lce_insert_every_n = CFG.mind.lce_insert_every_n,
        cerebellum_enabled = CFG.mind.cerebellum_enabled,
        pretrained         = CFG.mind.base_model,
    )

    params = model.param_count()
    print(f"   Model: {params['total']/1e6:.1f}M params | "
          f"LCE: {params['lce']/1e3:.0f}K | {model.model_size_mb():.0f}MB")

    # Print LCE info
    if model.lce_layers:
        print(f"   {model.lce_layers[0]}")

    # Train
    trainer = Trainer(model, tokenizer, CFG)
    trainer.train(train_ds, val_ds)

    # Save final model
    final_path = CFG.paths.checkpoints / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Model saved: {final_path}")

    return model, tokenizer


def stage_compress(model=None, tokenizer=None):
    print("\n" + "="*60)
    print("⑤ COMPRESSING (Body: prune + quantize)")
    print("="*60)

    if model is None:
        tokenizer = AmharicTokenizer()
        tokenizer.load(CFG.paths.root / "models" / "tokenizer")
        model = AmharicGPT2(
            vocab_size         = tokenizer.vocab_size_actual(),
            d_model            = CFG.mind.d_model,
            n_layers           = CFG.mind.n_layers,
            n_heads            = CFG.mind.n_heads,
            max_seq_len        = CFG.mind.max_seq_len,
            lce_enabled        = CFG.mind.lce_enabled,
            lce_latent_dim     = CFG.mind.lce_latent_dim,
            lce_insert_every_n = CFG.mind.lce_insert_every_n,
            cerebellum_enabled = CFG.mind.cerebellum_enabled,
            pretrained         = None,
        )
        ckpt = CFG.paths.checkpoints / "final_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    # Prune attention heads
    pruned = model.prune_attention_heads(sparsity=CFG.mind.attention_sparsity)
    n_pruned = sum(len(v) for v in pruned.values())
    print(f"   Pruned {n_pruned} attention heads ({CFG.mind.attention_sparsity*100:.0f}% sparsity)")

    # Quantize
    quantizer  = ModelQuantizer(method=CFG.body.quantize_method)
    model_q    = quantizer.quantize(model)
    comparison = quantizer.size_comparison(model, model_q)
    print(f"   Size: {comparison['original_mb']}MB → {comparison['quantized_mb']}MB "
          f"({comparison['reduction']} reduction)")

    return model_q, tokenizer


def stage_export(model=None, tokenizer=None):
    print("\n" + "="*60)
    print("⑥ EXPORTING TO MOBILE (Body: ONNX + bundle)")
    print("="*60)

    if model is None:
        model, tokenizer = stage_compress()

    exporter   = MobileExporter(CFG.paths.exported)
    onnx_path  = exporter.export_onnx(model, tokenizer)
    exporter.export_tflite(onnx_path)
    bundle     = exporter.create_bundle(onnx_path, tokenizer)
    print(f"✅ Mobile bundle ready: {bundle}")
    return bundle


def stage_test(model=None, tokenizer=None):
    print("\n" + "="*60)
    print("⑦ EVALUATING")
    print("="*60)

    if model is None:
        tokenizer = AmharicTokenizer()
        tokenizer.load(CFG.paths.root / "models" / "tokenizer")
        model = AmharicGPT2(
            vocab_size         = tokenizer.vocab_size_actual(),
            d_model            = CFG.mind.d_model,
            n_layers           = CFG.mind.n_layers,
            n_heads            = CFG.mind.n_heads,
            max_seq_len        = CFG.mind.max_seq_len,
            lce_enabled        = CFG.mind.lce_enabled,
            lce_latent_dim     = CFG.mind.lce_latent_dim,
            lce_insert_every_n = CFG.mind.lce_insert_every_n,
            cerebellum_enabled = CFG.mind.cerebellum_enabled,
            pretrained         = None,
        )
        ckpt = CFG.paths.checkpoints / "final_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    corpus_path = CFG.paths.data_clean / "corpus_final.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        sentences = [s for s in f.read().split("\n") if s.strip()]

    n       = len(sentences)
    val_s   = sentences[int(n*0.90):int(n*0.95)]
    test_s  = sentences[int(n*0.95):]

    val_ds  = AmharicDataset(val_s,  tokenizer, CFG.mind.max_seq_len)
    test_ds = AmharicDataset(test_s, tokenizer, CFG.mind.max_seq_len)

    evaluator = Evaluator(model, tokenizer, CFG)
    report    = evaluator.full_report(None, val_ds, test_ds)
    return report


def stage_chat():
    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT — Hasab (ሃሳብ)")
    print("="*60)
    print("Loading mobile runtime...")

    runtime = MobileRuntime(CFG.paths.exported)
    try:
        runtime.load()
    except FileNotFoundError:
        print("⚠️  No exported model found. Running full pipeline first...")
        stage_full()
        runtime.load()

    values  = CoreValues()
    persona = PersonaTracker()
    history = []

    print(f"\n{values.name} is ready. Type 'quit' to exit.\n")
    print("─" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nደህና ሁን!")
            break

        if user_input.lower() in ["quit", "exit", "ደህና ሁን"]:
            print("ደህና ሁን! 👋")
            break
        if not user_input:
            continue

        persona.update(user_input)
        response = runtime.chat(user_input, history)

        print(f"ሃሳብ: {response}\n")
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


def stage_full():
    print("\n🇪🇹 Amharic Mobile-AGI — Full Pipeline")
    print("Soul → Heart → Mind → Body")
    print("="*60)
    CFG.make_dirs()

    texts     = stage_clean()
    tokenizer = stage_tokenizer(texts)
    model, _  = stage_finetune(tokenizer)
    model_q, _= stage_compress(model, tokenizer)
    stage_export(model_q, tokenizer)
    stage_test(model, tokenizer)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print(f"   Model: {CFG.paths.exported}/hasab_mobile_bundle.zip")
    print(f"   Chat:  python main.py --mode chat")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Amharic Mobile-AGI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "acquire", "clean", "tokenizer",
                 "finetune", "compress", "export", "test", "chat"],
        default="full",
        help="Pipeline stage to run",
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total fine-tuning steps")
    parser.add_argument("--lce-warmup-steps", type=int, default=None,
                        help="Override LCE warmup steps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Batches to accumulate before an optimizer step")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers; 0 is safest in Colab")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Training device (default: auto-detect CUDA)")
    args = parser.parse_args()

    if args.max_steps is not None:
        CFG.training.max_steps = args.max_steps
    if args.lce_warmup_steps is not None:
        CFG.training.lce_warmup_steps = args.lce_warmup_steps
    if args.batch_size is not None:
        CFG.training.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        CFG.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_workers is not None:
        CFG.training.num_workers = args.num_workers
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("--device cuda was requested, but CUDA is not available")
        CFG.training.device = args.device

    stages = {
        "full":      stage_full,
        "acquire":   stage_acquire,
        "clean":     stage_clean,
        "tokenizer": stage_tokenizer,
        "finetune":  stage_finetune,
        "compress":  stage_compress,
        "export":    stage_export,
        "test":      stage_test,
        "chat":      stage_chat,
    }

    CFG.make_dirs()
    stages[args.mode]()


if __name__ == "__main__":
    main()
