import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256      # Maximum context length (in tokens)
    vocab_size: int = 4096     # Matches BPE tokenizer vocab_size in prepare.py
    n_layer: int = 6           # Number of Transformer decoder blocks
    n_head: int = 6            # Number of parallel self-attention heads
    n_embd: int = 384          # Embedding dimension (384 / 6 = 64 dims per head)
    dropout: float = 0.1       # Dropout probability for regularization
    bias: bool = False         # False is slightly faster and standard in modern LLMs


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention implemented in batched tensor form."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Combined Q, K, V projection for efficiency (3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back into the residual pathway
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check if PyTorch >= 2.0 Flash Attention is available
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Causal lower-triangular mask for manual attention fallback
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Time (seq_len), Channels (n_embd)

        # Compute Q, K, V in one matrix multiply and split into heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        if self.flash:
            # Hardware-accelerated FlashAttention / Memory-Efficient Attention
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual scaled dot-product causal attention (Karpathy's explicit math)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hd) -> (B, nh, T, hd)

        # Re-assemble all head outputs side by side -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP: expands 4x, applies GELU, projects back to n_embd."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer Decoder Block: Communication (Attention) + Computation (MLP) with Pre-Norm."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections allow unimpeded gradient flow during backpropagation
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoCoderGPT(nn.Module):
    """Decoder-Only GPT Language Model for Python & Ubuntu CLI Autocompletion."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),      # Token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),      # Positional embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),      # Final LayerNorm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between input token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights cleanly
        self.apply(self._init_weights)
        # Apply special scaled initialization to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Forward pass through embeddings + decoder blocks
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only compute logits for the very last token
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with Temperature scaling & Top-K filtering."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional Top-K filtering to prevent sampling bizarre low-probability tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop early if end-of-document token is generated
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

        return idx


if __name__ == "__main__":
    # Quick sanity check when running `python3 src/model.py`
    config = GPTConfig()
    model = NanoCoderGPT(config)
    print(f"NanoCoderGPT initialized successfully | Parameters: {model.get_num_params() / 1e6:.2f}M")

    dummy_x = torch.randint(0, config.vocab_size, (2, 32))
    dummy_y = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(dummy_x, dummy_y)
    print(f"Forward pass OK | Logits shape: {logits.shape} | Initial loss: {loss.item():.4f}")