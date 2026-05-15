# 🇪🇹 Hasab (ሃሳብ) — Google Colab Setup Guide

This guide will walk you through setting up and training the **Hasab Amharic AGI** model on a free Google Colab GPU. 

Because the project includes an automated data pipeline, custom tokenizer, and ONNX mobile export, Google Colab's free **T4 GPU** is perfect for running the entire workflow from end to end.

---

## Step 1: Open Google Colab & Enable GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click on **New Notebook**.
3. In the top menu, go to **Runtime > Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU**.
5. Click **Save**.

---

## Step 2: Clone the Repository

In the first cell of your Colab notebook, clone your GitHub repository and navigate into the project folder.

```python
!git clone https://github.com/ethcocoder/hasab-ai.git
%cd hasab-ai
```

---

## Step 3: Install Dependencies

Install the required Python packages for the pipeline. This includes PyTorch, Transformers, ONNX Runtime (for mobile export), and Wikipedia-API (for Amharic corpus acquisition).

Run this in a new cell:

```python
!pip install -r requirements.txt
!pip install Wikipedia-API onnx onnxruntime sentencepiece -q
```

---

## Step 4: Run the Training Pipeline

You have two options for running the pipeline: using the provided Jupyter Notebook or using the command-line interface (`main.py`).

### Option A: Using the Interactive Notebook (Recommended)
We have already included a fully configured notebook (`Amharic_AGI_Training.ipynb`) inside the repository.
1. In Colab, click **File > Open notebook**.
2. Select the **GitHub** tab.
3. Paste your repository URL: `https://github.com/ethcocoder/hasab-ai.git`
4. Open the `Amharic_AGI_Training.ipynb` file.
5. Run the cells one by one.

### Option B: Using the CLI (`main.py`)
If you prefer to just launch the entire automated pipeline at once, you can run the main orchestrator script directly:

```python
# Run the full pipeline (Acquire -> Clean -> Tokenizer -> Train -> Compress -> Export -> Test)
!python main.py --mode full
```

*Note: The full training on a T4 GPU usually takes roughly 2 to 4 hours depending on the `max_steps` set in `config.py`.*

---

## Step 5: Test the Interactive Chat

Once the pipeline finishes and the ONNX model is exported, you can test the chatbot's runtime environment directly inside Colab:

```python
!python main.py --mode chat
```
*(Type "quit" to exit the chat)*

---

## Step 6: Download the Mobile Bundle

The ultimate goal of this pipeline is to produce a lightweight, compressed `.zip` bundle that can be deployed on an Android or iOS device. 

Once training and export are complete, run this cell to download the mobile bundle to your local computer:

```python
from google.colab import files

# Download the final quantized ONNX bundle
files.download('models/exported/hasab_mobile_bundle.zip')
```

---

### 💡 Troubleshooting Colab OOM (Out of Memory) Errors
If you run out of GPU memory during the `finetune` stage, you need to adjust the batch size. 
Open `config.py` in Colab (using the file explorer on the left) and change:
```python
CFG.training.batch_size = 4  # Default is 8, reduce this if you hit OOM limits
```
You can also reduce the `max_steps` for a faster, initial test run.
