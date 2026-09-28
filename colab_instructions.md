# Hasab (ሃሳብ) — Google Colab GPU terminal guide

This project is designed to run from the **Colab terminal**. The Jupyter notebook has been removed.

## 1. Create a GPU runtime

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Choose **Runtime → Change runtime type → T4 GPU** (or another available NVIDIA GPU).
3. Open the terminal with **File → Open terminal**.
4. Verify the GPU:

```bash
nvidia-smi
```

If `nvidia-smi` does not show a GPU, return to step 2 before installing anything.

## 2. Clone the project and install dependencies

Run these commands in the Colab terminal:

```bash
cd /content
git clone https://github.com/ethcocoder/hasab-ai.git
cd hasab-ai
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins `transformers` below 5.0 because this project inserts
custom LCE layers into GPT-2's internal blocks. If Transformers 5 was already
installed in the runtime, the command above will downgrade it to the supported
4.x release.

Colab normally includes a CUDA-enabled PyTorch build. Confirm that Python can see it:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

The project downloads the pretrained `gpt2` weights the first time `finetune` runs. Make sure the runtime has internet access and enough disk space.

## 3. Run a small smoke test first

Do not start with 50,000 steps. First verify every stage with a short run:

```bash
python main.py --mode full \
  --max-steps 20 \
  --lce-warmup-steps 5 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --device cuda
```

This command runs:

1. data acquisition from Amharic Wikipedia;
2. cleaning and safety filtering;
3. tokenizer training;
4. GPU fine-tuning;
5. pruning and dynamic INT8 quantization;
6. ONNX export and mobile bundle creation; and
7. evaluation.

The first run may take time while GPT-2 and Python packages are downloaded.

## 4. Start a real training run

After the smoke test succeeds, choose settings based on the available GPU memory. A T4 usually has 16 GB VRAM:

```bash
python main.py --mode full \
  --max-steps 10000 \
  --lce-warmup-steps 500 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-workers 0 \
  --device cuda
```

The effective batch size is:

```text
batch-size × gradient-accumulation-steps
```

For the original default-length run, omit `--max-steps` and `--lce-warmup-steps`, but expect a long run:

```bash
python main.py --mode full --batch-size 2 --gradient-accumulation-steps 8 --num-workers 0 --device cuda
```

### If you receive CUDA out-of-memory errors

Retry with a smaller per-device batch and/or sequence length. The sequence length is configured in `config.py`:

```bash
python main.py --mode full \
  --max-steps 10000 \
  --lce-warmup-steps 500 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-workers 0 \
  --device cuda
```

Then, if necessary, edit `CFG.mind.max_seq_len` in `config.py` from `512` to `256`. Keep the change consistent for the whole run.

## 5. Run individual stages from the terminal

You can rerun a stage without rerunning the entire pipeline:

```bash
python main.py --mode acquire
python main.py --mode clean
python main.py --mode tokenizer
python main.py --mode finetune --max-steps 10000 --batch-size 2 --num-workers 0 --device cuda
python main.py --mode compress
python main.py --mode export
python main.py --mode test
python main.py --mode chat
```

Use `python main.py --help` to see all options.

## 6. Keep outputs after the Colab session ends

Colab runtimes are temporary. Copy the repository outputs to Google Drive before disconnecting. In the terminal:

```bash
mkdir -p /content/drive/MyDrive/hasab-ai-output
cp -r models logs /content/drive/MyDrive/hasab-ai-output/
```

If you want the source and outputs to persist between sessions, clone the repository into Drive instead:

```bash
git clone https://github.com/ethcocoder/hasab-ai.git /content/drive/MyDrive/hasab-ai
cd /content/drive/MyDrive/hasab-ai
```

## 7. Expected output files

After a successful run, the important files are:

```text
models/checkpoints/final_model.pt
models/tokenizer/tokenizer.json
models/tokenizer/tokenizer_config.json
models/exported/hasab_amharic.onnx
models/exported/hasab_mobile_bundle.zip
logs/eval_report.json
```

The ONNX runtime now prefers the serialized BPE tokenizer from `tokenizer.json`, so terminal chat and exported inference use the same tokenization as training.

## 8. Troubleshooting

- **`CUDA is not available`**: select a GPU runtime and restart the runtime.
- **`No raw files found`**: the pipeline automatically runs acquisition; check internet access if Wikipedia download fails.
- **Hugging Face download errors**: rerun `python main.py --mode finetune` after confirming internet access.
- **Colab disconnects**: use fewer steps, save checkpoints, and copy `models/` to Drive.
- **Export problems related to TFLite**: the pipeline treats TFLite as optional. ONNX export and the mobile bundle are the primary outputs.
