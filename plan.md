 🗂️ Project Layout — Token Importance Dropout (TID)

This is the full file structure and purpose of each component used in the TID research project. Designed to run efficiently on a single RTX 4060 8GB using mixed precision and TF32 where available.

---

## 📁 `data/`
Contains preprocessed, **streamed + limited** dataset files (OpenWebText).

- `train.jsonl`: 10,000 streamed samples of OpenWebText (text only), each line = raw string.
- `val.jsonl`: 1,000 samples (held-out subset).

> **Important**: Load OpenWebText with HuggingFace streaming mode.  
> Never download full dataset. Limit to 10,000 samples via `islice`.

---

## 📁 `tokenizer/`
Contains the tokenizer config and cache.

- `tokenizer.json`: GPT-2 tokenizer (from `tiktoken`) serialized for reuse.

---

## 📁 `src/`

### `token_dropout.py`
- core module: handles **importance scoring** + **dropout masking**
- defines multiple scoring methods:  
  - gradient norm (uses `.retain_grad()`)
  - entropy of softmaxed logits
  - attention mass (optional, more compute)
- dropout applied *before* MHA/FFN (zero out embeddings)

---

### `model.py`
- minimal GPT-2 variant (125M), modified:
  - RoPE embeddings
  - RMSNorm
  - SwiGLU
  - **No attention bias**
- support for mixed precision + TF32 (autocast, `torch.set_float32_matmul_precision('high')`)
- no position bias to reduce distractions from position priors

---

### `load_data.py`
- loads **OpenWebText** via HuggingFace datasets (streaming=True)
- truncates to 10k examples for train, 1k for val
- saves as `jsonl` (text only)
- uses `tiktoken` GPT-2 tokenizer
- chunk to `context=128` with optional overlapping windows

---

### `train.py`
- main training loop
- supports mixed precision via `torch.cuda.amp.autocast` + `GradScaler`
- loads model, optimizer (AdamW), cosine LR scheduler
- applies `TokenImportanceDropout` during training only
- logs BPC, val loss, step time

---

### `evaluate.py`
- compute bits-per-character (BPC) and val loss on `val.jsonl`
- disables dropout
- uses same batching/precision as training

---

### `config.py`
central config file with everything:

```python
MODEL_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
VOCAB_SIZE = 50257
CONTEXT_LEN = 128
BATCH_SIZE = 8
TOTAL_STEPS = 200_000
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
DATA_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
DEVICE = "cuda"
USE_MIXED_PRECISION = True
USE_TOKEN_DROPOUT = True
DROP_SCHEDULE = "linear_decay"

📁 logs/

    stores .csv or .json logs of val loss, BPC, dropout ratio over time

📁 checkpoints/

    contains model checkpoints (.pt) every N steps

    last + best model both saved

🧩 important details

    TF32: enabled globally with torch.set_float32_matmul_precision('high')

    AMP: enabled in training + eval using autocast() and GradScaler

    Streaming Dataset: use datasets.load_dataset("openwebtext", streaming=True) with islice

    Limit: enforce hard limit of 10k samples during streaming (train) and 1k (val)

    Memory-aware: model fits easily in 8GB VRAM with context=128 and 4-bit optimizer if needed