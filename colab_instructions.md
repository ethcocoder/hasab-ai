# Hasab (ሃሳብ) — Google Colab GPU terminal guide

Hasab now uses **Qwen2.5-0.5B with LoRA** as its only language-model backend. The notebook workflow has been removed.

## 1. Create a GPU runtime

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Open **File → Open terminal**.
4. Verify the GPU:

```bash
nvidia-smi
```

## 2. Clone and install

```bash
cd /content
git clone https://github.com/ethcocoder/hasab-ai.git
cd hasab-ai
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If importing `peft` fails with `operator torchvision::nms does not exist`,
Colab's optional torchvision package does not match its PyTorch package. Run
this once, then restart the Colab runtime and reinstall the project packages:

```bash
python -m pip uninstall -y torchvision
python -m pip install --no-cache-dir --force-reinstall \
  "transformers==4.46.3" "peft==0.13.2" "accelerate==0.34.2"
```

After restarting the runtime, rerun the installation command above. Do not
reinstall `torch` manually; Colab already provides the CUDA-enabled build.

The project uses `transformers<5` for compatibility with the current PEFT/Qwen integration. Confirm the environment:

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python - <<'PY'
import torch, transformers, peft
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

The `USE_TF=0` setting is intentional. Qwen fine-tuning uses PyTorch only;
loading Colab's TensorFlow stack can cause a native-library segmentation fault.

The first training run downloads `Qwen/Qwen2.5-0.5B`. It fits on a T4 when trained with LoRA.

## 3. Run the Qwen smoke test

Your current Wikipedia run produces only a few hundred sentences, so first build
the quality-controlled datasets:

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python main.py --mode prepare-data
```

This creates:

```text
data/processed/knowledge.jsonl
data/processed/chat.jsonl
data/processed/review.jsonl
data/processed/manifest.json
```

Inspect the manifest before training:

```bash
cat data/processed/manifest.json
```

The builder removes short lines, headings, URL records, malformed fragments,
near-duplicates, low-Amharic records, and unsafe records. It flags numeric,
historical, and political claims for human review. These records are held out
of training and written to `review.jsonl`, so unverified facts are not taught
to the model by default.

Then run the Qwen smoke test:

```bash
python main.py --mode full \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 256 \
  --num-workers 0 \
  --device cuda
```

This runs:

1. Amharic data acquisition;
2. quality-controlled knowledge and chat dataset preparation;
3. Qwen tokenizer loading;
4. Qwen2.5-0.5B LoRA fine-tuning using a 70/30 knowledge/chat mixture; and
5. sample generation evaluation using Qwen's official chat template.

The adapter is saved to:

```text
models/qwen_adapter/
```

The base Qwen model remains in the Hugging Face cache. Only the small LoRA adapter needs to be copied as your project output.

## 4. Test chat after training

```bash
python main.py --mode chat --device cuda
```

Type `quit` to exit.

You can also run evaluation without interactive chat:

```bash
python main.py --mode test --device cuda
```

## 5. Start a longer LoRA run

After the smoke test completes successfully:

```bash
python main.py --mode full \
  --max-steps 2000 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 512 \
  --num-workers 0 \
  --device cuda
```

The effective batch size is:

```text
batch-size × gradient-accumulation-steps
```

Because the initial corpus is small, collect more Amharic text before treating a longer run as a quality experiment. A useful next target is at least **100,000 Amharic sentences**.

## 6. If you get CUDA out-of-memory errors

Use a shorter sequence and keep batch size at 1:

```bash
python main.py --mode full \
  --max-steps 1000 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 128 \
  --num-workers 0 \
  --device cuda
```

Do not replace Qwen's tokenizer with the old custom tokenizer. Qwen must use its original tokenizer because its vocabulary and embedding IDs are pretrained together.

## 7. Run individual stages

```bash
python main.py --mode acquire
python main.py --mode prepare-data
python main.py --mode finetune --max-steps 1000 --batch-size 1 --num-workers 0 --device cuda
python main.py --mode test --device cuda
python main.py --mode chat --device cuda
```

Use `python main.py --help` for all options.

## 8. Persist the adapter in Google Drive

```bash
mkdir -p /content/drive/MyDrive/hasab-ai-output
cp -r models/qwen_adapter /content/drive/MyDrive/hasab-ai-output/
cp -r data/processed /content/drive/MyDrive/hasab-ai-output/
```

Important output files:

```text
models/qwen_adapter/adapter_config.json
models/qwen_adapter/adapter_model.safetensors
models/qwen_adapter/tokenizer.json
models/qwen_adapter/tokenizer_config.json
data/processed/knowledge.jsonl
data/processed/chat.jsonl
data/processed/manifest.json
```

## Troubleshooting

- **`CUDA is not available`**: select a GPU runtime and restart Colab.
- **Transformers version errors**: run `python -m pip install --upgrade --force-reinstall -r requirements.txt`.
- **Hugging Face download errors**: verify internet access and rerun `python main.py --mode finetune`.
- **Out of memory**: use `--batch-size 1 --max-length 128`.
- **No adapter found**: run `python main.py --mode finetune` before `test` or `chat`.
