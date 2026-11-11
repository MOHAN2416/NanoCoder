import os
import sys
import time
import torch
from tokenizers import Tokenizer

# Allow importing model from src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import GPTConfig, NanoCoderGPT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "ckpt.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "processed", "tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_nanocoder():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}.\n"
            "Train the model first (via train.py or Google Colab) and place ckpt.pt in checkpoints/."
        )

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    config = GPTConfig(**checkpoint["config"])

    model = NanoCoderGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def complete_code(
    model: NanoCoderGPT,
    tokenizer: Tokenizer,
    prompt: str,
    mode: str = "python",
    max_tokens: int = 80,
    temperature: float = 0.4,
    top_k: int = 40,
):
    prefix = "<|python|>\n" if mode == "python" else "<|bash|>\n"
    full_prompt = prefix + prompt

    input_ids = tokenizer.encode(full_prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    t0 = time.perf_counter()
    output_ids = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_id,
    )
    dt = time.perf_counter() - t0

    new_tokens = output_ids[0].tolist()[len(input_ids) :]
    completion = tokenizer.decode(new_tokens)

    # Clean up trailing special token if present
    completion = completion.replace("<|endoftext|>", "").rstrip()
    tok_per_sec = len(new_tokens) / max(dt, 1e-6)
    return completion, len(new_tokens), dt * 1000, tok_per_sec


def interactive_cli():
    print("Loading NanoCoder...")
    model, tokenizer = load_nanocoder()
    print(f"NanoCoder ready on {DEVICE.upper()} ({model.get_num_params() / 1e6:.2f}M params)")
    print("Commands: '/py' (Python mode), '/sh' (Ubuntu Bash mode), '/exit' to quit\n")

    mode = "python"
    while True:
        try:
            tag = "PY" if mode == "python" else "BASH"
            user_input = input(f"[{tag}] >>> ")

            if not user_input.strip():
                continue
            if user_input.strip() == "/exit":
                break
            if user_input.strip() == "/py":
                mode = "python"
                print("Switched to Python autocomplete mode.\n")
                continue
            if user_input.strip() == "/sh":
                mode = "bash"
                print("Switched to Ubuntu CLI/Bash autocomplete mode.\n")
                continue

            completion, n_tok, ms, tps = complete_code(model, tokenizer, user_input, mode=mode)
            print(f"\033[92m{user_input}\033[96m{completion}\033[0m")
            print(f"\033[90m[{n_tok} tokens | {ms:.1f} ms | {tps:.1f} tok/s]\033[0m\n")

        except KeyboardInterrupt:
            print("\nExiting NanoCoder.")
            break


if __name__ == "__main__":
    interactive_cli()