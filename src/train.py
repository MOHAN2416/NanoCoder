import os
import time
import math
import json
import numpy as np
import torch
from model import GPTConfig, NanoCoderGPT

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Training Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_ITERS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Device & Mixed Precision Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)


def get_batch(split: str):
    """Samples a random batch of input-target chunks using memory-mapped arrays."""
    file_name = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(PROCESSED_DIR, file_name), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix])

    if DEVICE == "cuda":
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def get_lr(step: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / (WARMUP_ITERS + 1)
    if step > MAX_ITERS:
        return MIN_LR
    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss(model: NanoCoderGPT, ctx):
    """Evaluates average loss across train and validation splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(1337)

    print(f"Training NanoCoder on device: {DEVICE.upper()} | Precision: {DTYPE}")

    config = GPTConfig(block_size=BLOCK_SIZE)
    model = NanoCoderGPT(config).to(DEVICE)
    print(f"Model Parameters: {model.get_num_params() / 1e6:.2f}M")

    # Separate parameters for weight decay (decay 2D weights, skip 1D biases/LayerNorms)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    history = []
    t0 = time.time()

    for step in range(MAX_ITERS + 1):
        # Update learning rate for current step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and save best checkpoint periodically
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx)
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"Step {step:4d}/{MAX_ITERS} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | Time: {dt:.1f}s"
            )
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr})

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, os.path.join(CKPT_DIR, "ckpt.pt"))
                print(f"  -> Saved new best checkpoint (val_loss: {best_val_loss:.4f})")

        if step == MAX_ITERS:
            break

        # Forward + Backward + Optimizer step
        xb, yb = get_batch("train")
        with ctx:
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

    # Save training history for plotting loss curves later
    with open(os.path.join(CKPT_DIR, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()