# NanoCoder

A **12.29M-parameter decoder-only GPT Transformer** built completely from scratch in PyTorch for local, offline Python code and Ubuntu CLI/Bash autocompletion[cite: 2, 3].

---

## Purpose & Motivation

Modern code assistants rely on massive cloud-hosted models that require constant internet connectivity, high API costs, and significant compute overhead. **NanoCoder** was built to:

1. **Demystify Transformer Internals:** Implement the core autoregressive Transformer architecture—Causal Multi-Head Self-Attention, Pre-LayerNorm, residual pathways, weight tying, and BPE subword tokenization—from first principles in PyTorch without relying on high-level `transformers` model abstractions.
2. **Enable Offline Local Autocompletion:** Train a compact domain-specific language model capable of running locally on a standard consumer CPU with sub-25ms per-token latency.
3. **Explore Bilingual Syntax Modeling:** Evaluate how a single small-scale decoder handles two distinct formal grammars—indentation-sensitive **Python** and flag-driven **Ubuntu Bash/CLI**—using explicit domain control tokens (`<|python|>` and `<|bash|>`).

---

## Architecture & Implementation Details

The model follows a GPT-style decoder-only architecture (`src/model.py`):

| Component | Specification |
| :--- | :--- |
| **Model Parameters** | `12.29M` (with tied input/output embedding weights)[cite: 1, 2] |
| **Layers (`n_layer`)** | `6` Transformer Decoder Blocks |
| **Attention Heads (`n_head`)** | `6` Causal Self-Attention Heads (`64` dims per head) |
| **Embedding Dimension (`n_embd`)** | `384` |
| **Context Window (`block_size`)** | `256` tokens |
| **Vocabulary Size (`vocab_size`)** | `4,096` (Custom Byte-Level BPE)[cite: 1] |
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
│   ├── prepare.py             # Corpus collector (Python stdlib + Ubuntu CLI scripts) & BPE trainer
│   ├── model.py               # Decoder-only GPT architecture & Temperature/Top-K sampler
│   └── train.py               # Training loop with cosine LR decay, warmup, & mixed precision
├── checkpoints/
│   └── training_log.json      # Step-by-step loss and learning rate telemetry
├── cli.py                     # Interactive terminal autocomplete REPL (/py and /sh modes)
├── evaluate.py                # Automated Python AST, Bash syntax, and CPU latency benchmark
├── plot_loss.png              # Training vs. validation loss curve
└── README.md