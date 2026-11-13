
# NanoCoder

A **12.29M-parameter decoder-only GPT Transformer** built completely from scratch in PyTorch for local, offline Python code and Ubuntu CLI/Bash autocompletion.

---

## Purpose & Motivation

Modern code assistants rely on massive cloud-hosted models that require constant internet connectivity, high API costs, and significant compute overhead.

**NanoCoder** was built to:

1. **Demystify Transformer Internals:** Implement the core autoregressive Transformer architecture — Causal Multi-Head Self-Attention, Pre-LayerNorm, residual pathways, weight tying, and BPE subword tokenization — from first principles in PyTorch without relying on high-level `transformers` model abstractions.

2. **Enable Offline Local Autocompletion:** Train a compact domain-specific language model capable of running locally on a standard consumer CPU with sub-25ms per-token latency.

3. **Explore Bilingual Syntax Modeling:** Evaluate how a single small-scale decoder handles two distinct formal grammars — indentation-sensitive **Python** and flag-driven **Ubuntu Bash/CLI** — using explicit domain control tokens (`<|python|>` and `<|bash|>`).

---

## Architecture & Implementation Details

The model follows a GPT-style decoder-only architecture (`src/model.py`).

| Component | Specification |
| :--- | :--- |
| **Model Parameters** | `12.29M` (with tied input/output embedding weights) |
| **Layers (`n_layer`)** | `6` Transformer Decoder Blocks |
| **Attention Heads (`n_head`)** | `6` Causal Self-Attention Heads (`64` dims per head) |
| **Embedding Dimension (`n_embd`)** | `384` |
| **Context Window (`block_size`)** | `256` tokens |
| **Vocabulary Size (`vocab_size`)** | `4,096` (Custom Byte-Level BPE) |
| **Normalization & Activation** | Pre-LayerNorm (`LayerNorm`) + `GELU` MLP (`4x` expansion) |
| **Attention Backend** | PyTorch `scaled_dot_product_attention` (FlashAttention) with manual causal mask fallback |
| **Dropout** | `0.1` |

### Repository Structure

```text
NanoCoder/
├── data/
│   └── processed/
│       ├── tokenizer.json     # Trained 4,096-token Byte-Level BPE tokenizer
│       ├── train.bin          # 90% training split (uint16 memory-mapped binary)
│       └── val.bin            # 10% validation split (uint16 memory-mapped binary)
├── src/
│   ├── prepare.py             # Corpus collector and BPE trainer
│   ├── model.py               # Decoder-only GPT architecture and sampler
│   └── train.py               # Training loop with cosine LR decay and warmup
├── checkpoints/
│   └── training_log.json      # Step-by-step loss and learning rate telemetry
├── cli.py                     # Interactive terminal autocomplete REPL (/py and /sh modes)
├── evaluate.py                # Python AST, Bash syntax, and CPU latency benchmarks
├── plot_loss.png              # Training vs. validation loss curve
└── README.md
```

---

## Training Setup & Dynamics

### Dataset Preparation

The training corpus was curated locally from:

- Python standard library and installed site-packages.
- Ubuntu system shell completion scripts:
  - `/usr/share/bash-completion`
  - `/etc/init.d`
  - `/usr/share/initramfs-tools`

The collected corpus was tokenized using a custom **4,096-token Byte-Level BPE tokenizer** and divided into:

- **90%** training data.
- **10%** validation data.

### Training Configuration

| Parameter | Configuration |
| :--- | :--- |
| **Training Hardware** | Kaggle NVIDIA T4 GPU |
| **Precision** | PyTorch `torch.bfloat16` automatic mixed precision |
| **Optimizer** | AdamW |
| **Optimizer Betas** | `(0.9, 0.95)` |
| **Weight Decay** | `0.1` on 2D weights |
| **Gradient Clipping** | `1.0` |
| **Training Iterations** | `5,000` |
| **Batch Size** | `32` |
| **Learning Rate** | `6e-4` |
| **Minimum Learning Rate** | `6e-5` |
| **Warmup Steps** | `200` |
| **Learning Rate Schedule** | Linear warmup followed by cosine decay |

### Training Results

| Metric | Value |
| :--- | :--- |
| **Initial Training Loss (Step 0)** | `8.3789` |
| **Initial Validation Loss (Step 0)** | `8.3797` |
| **Final Training Loss (Step 5000)** | `2.0731` |
| **Best Validation Loss** | `4.1002` (Step 4750) |

The initial loss is close to the theoretical random prediction baseline:

\[
-\ln(1/4096) \approx 8.32
\]

### Note on the Training/Validation Gap

The training corpus was concatenated sequentially, with Python files followed by Ubuntu shell scripts, before the 90/10 split.

As a result, the validation split contains a higher concentration of shell scripts than the training split.

This introduces domain skew between the training and validation datasets and likely contributes to the observed loss gap.

**Potential improvement:** Shuffle documents before splitting the dataset to ensure a more balanced distribution of Python and Bash samples across both sets.

---

## Evaluation & Benchmarks

All benchmarks were measured locally on a **CPU-only laptop** using `evaluate.py`.

The evaluation covered **40 test prompts**:

- 20 Python prompts validated using Python's `ast.parse()`.
- 20 Ubuntu Bash prompts validated using static `bash -n` syntax checking.

### Benchmark Results

| Metric | Measured Value |
| :--- | :--- |
| **Total Trainable Parameters** | `12.29M` |
| **Best Validation Loss** | `4.1002` |
| **Validation Perplexity** | `60.35` |
| **Python AST Syntax Validity** | `45.0%` (9 / 20 valid completions) |
| **Ubuntu Bash Syntax Validity** | `60.0%` (12 / 20 valid completions) |
| **Local CPU Inference Latency** | `21.48 ms / token` |
| **Local CPU Throughput** | `46.6 tokens/sec` |

### Sample Local CPU Output

The following is an example of NanoCoder generating Python code through `cli.py`.

**Configuration:**
- Mode: Python (`/py`)
- Temperature: `0.4`
- Top-K: `40`

**Prompt:**

```python
def read_json_file(filepath):
```

**Generated Output:**

```python
def read_json_file(filepath):
    """
    Return a filepath for a filepath.

    The filepath is a filepath.

    The filepath is a filepath.
    """
    filepath = filepath.join(filepath, filepath)
    if filepath.exists(filepath):
        return filepath
```

```text
# [80 tokens | 1773.7 ms | 45.1 tok/s]
```

Despite its limited parameter budget of 12.29M, the model demonstrates the ability to infer:

- Four-space Python indentation.
- Docstring delimiters.
- Conditional block structures.
- Basic Python syntax patterns.

The generated code is not necessarily semantically correct. Syntax learning and functional code generation are two very different problems.

---

## Getting Started

### 1. Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Mohan2416/NanoCoder.git

cd NanoCoder
```

Create and activate a Python virtual environment:

```bash
python3 -m venv venv

source venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch numpy tokenizers tqdm matplotlib
```

### 2. Build Dataset & Tokenizer

Collect Python and Ubuntu CLI scripts from your local system and encode them into `train.bin` and `val.bin`.

```bash
python3 src/prepare.py
```

### 3. Train the Model

Start model training:

```bash
python3 src/train.py
```

The training script automatically uses CUDA with bfloat16/float16 mixed precision when a compatible GPU is available, or falls back to CPU.

### 4. Run Benchmarks & Interactive CLI

Once the model has been trained and `checkpoints/ckpt.pt` has been generated, run the evaluation:

```bash
python3 evaluate.py
```

Launch the interactive terminal autocompleter:

```bash
python3 cli.py
```

### CLI Commands

Inside the interactive CLI:

| Command | Description |
| :--- | :--- |
| `/py` | Switch to Python autocomplete mode |
| `/sh` | Switch to Ubuntu Bash autocomplete mode |
| `/exit` | Exit the interactive terminal |

---

## Future Improvements

### 1. Document-Level Shuffling

Shuffle Python and Bash documents before performing the 90/10 training-validation split.

This will eliminate domain skew between the two datasets and provide a more representative validation metric.

### 2. Repetition Penalty & Fill-in-the-Middle (FIM)

Introduce repetition penalties during decoding to reduce repetitive and redundant generations.

Add Fill-in-the-Middle special tokens:

- `<|fim_prefix|>`
- `<|fim_middle|>`
- `<|fim_suffix|>`

This would enable NanoCoder to support mid-line cursor completion rather than relying exclusively on left-to-right generation.

### 3. Modern Architectural Primitives

Experiment with modern Transformer components and benchmark them against the current GPT-2-style baseline:

- **RoPE:** Rotary Position Embeddings.
- **RMSNorm:** Root Mean Square Layer Normalization.
- **SwiGLU:** Swish-Gated Linear Unit activation.

The objective is to evaluate whether these architectural changes improve generation quality and training efficiency within the same parameter budget.

---

## Project Summary

**NanoCoder** is an exploration of building a compact, locally executable code language model entirely from scratch.

With only **12.29M parameters**, it demonstrates:

- A complete GPT-style decoder-only Transformer implemented in PyTorch.
- Custom Byte-Level BPE tokenization.
- Domain-specific Python and Ubuntu Bash generation.
- GPU-based training and CPU-based inference.
- Offline terminal-based code autocompletion.
- Automated syntax validation and performance benchmarking.

The project serves as both a practical experiment in small language models and a hands-on exploration of Transformer architecture, tokenizer design, model training, and local inference.
