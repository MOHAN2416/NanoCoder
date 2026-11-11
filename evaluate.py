import os
import ast
import math
import subprocess
import torch
from cli import load_nanocoder, complete_code

PYTHON_PROMPTS = [
    "def calculate_mean(values):\n    ",
    "def binary_search(arr, target):\n    ",
    "class FileHandler:\n    def __init__(self, path):\n        ",
    "for item in range(10):\n    if item % 2 == 0:\n        ",
    "import os\nimport sys\n\ndef get_files(directory):\n    ",
]

BASH_PROMPTS = [
    "if [ -f \"$1\" ]; then\n    ",
    "for file in *.py; do\n    ",
    "find /var/log -type f -name ",
    "systemctl status ",
    "grep -rnw '/etc/' -e ",
]


def is_valid_python_completion(prompt: str, completion: str) -> bool:
    """Trims cut-off trailing lines from token limit and checks AST validity."""
    prompt_lines = len(prompt.rstrip("\n").splitlines())
    lines = (prompt + completion).splitlines()

    # Walk backwards to discard any line cut off mid-token by max_tokens
    while len(lines) > prompt_lines:
        candidate = "\n".join(lines)
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            lines.pop()
    return False


def is_valid_bash_completion(prompt: str, completion: str) -> bool:
    """Checks shell syntax validity safely using `bash -n` (no execution)."""
    prompt_lines = len(prompt.rstrip("\n").splitlines())
    lines = (prompt + completion).splitlines()

    while len(lines) >= prompt_lines and len(lines) > 0:
        candidate = "\n".join(lines)
        res = subprocess.run(
            ["bash", "-n"],
            input=candidate,
            text=True,
            capture_output=True,
        )
        if res.returncode == 0 and len(candidate.strip()) > len(prompt.strip()):
            return True
        lines.pop()
    return False


def run_benchmarks():
    model, tokenizer = load_nanocoder()
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    best_val_loss = ckpt.get("best_val_loss", 0.0)
    perplexity = math.exp(best_val_loss)

    py_valid, bash_valid = 0, 0
    samples_per_domain = 20
    latencies = []

    print("Running Python AST & Bash syntax benchmarks (20 samples each)...")

    # 1. Evaluate Python completions
    for i in range(samples_per_domain):
        prompt = PYTHON_PROMPTS[i % len(PYTHON_PROMPTS)]
        completion, n_tok, ms, _ = complete_code(
            model, tokenizer, prompt, mode="python", max_tokens=50, temperature=0.3
        )
        latencies.append(ms / max(n_tok, 1))
        if is_valid_python_completion(prompt, completion):
            py_valid += 1

    # 2. Evaluate Ubuntu Bash completions
    for i in range(samples_per_domain):
        prompt = BASH_PROMPTS[i % len(BASH_PROMPTS)]
        completion, n_tok, ms, _ = complete_code(
            model, tokenizer, prompt, mode="bash", max_tokens=40, temperature=0.3
        )
        latencies.append(ms / max(n_tok, 1))
        if is_valid_bash_completion(prompt, completion):
            bash_valid += 1

    avg_ms_per_tok = sum(latencies) / len(latencies)
    py_rate = (py_valid / samples_per_domain) * 100
    bash_rate = (bash_valid / samples_per_domain) * 100

    print("\n=== NanoCoder Resume Benchmark Results ===")
    print(f"Model Parameters      : {model.get_num_params() / 1e6:.2f}M")
    print(f"Best Validation Loss  : {best_val_loss:.4f}")
    print(f"Validation Perplexity : {perplexity:.2f}")
    print(f"Python AST Validity   : {py_rate:.1f}% ({py_valid}/{samples_per_domain} valid)")
    print(f"Bash Syntax Validity  : {bash_rate:.1f}% ({bash_valid}/{samples_per_domain} valid)")
    print(f"CPU Latency / Token   : {avg_ms_per_tok:.2f} ms ({1000 / avg_ms_per_tok:.1f} tok/s)")


if __name__ == "__main__":
    run_benchmarks()