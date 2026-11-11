import os
import glob
import sysconfig
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Resolve project root dynamically (works whether run from root or src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "raw", "code_cli_corpus.txt")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VOCAB_SIZE = 4096

# Dynamically detect active Python stdlib & site-packages
PYTHON_DIRS = [
    sysconfig.get_paths()["stdlib"],
    sysconfig.get_paths()["purelib"],
]

# Ubuntu system directories containing clean Bash/CLI scripts
BASH_DIRS = [
    "/usr/share/bash-completion",
    "/etc/init.d",
    "/usr/share/initramfs-tools",
]

def collect_corpus():
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Gather Python files
    py_files = []
    for base_dir in set(PYTHON_DIRS):
        if base_dir and os.path.exists(base_dir):
            py_files.extend(glob.glob(f"{base_dir}/**/*.py", recursive=True))

    clean_py = [
        f for f in py_files
        if "test" not in f.lower() and os.path.getsize(f) < 100_000
    ]

    # 2. Gather Ubuntu Bash / CLI scripts
    sh_files = []
    for base_dir in BASH_DIRS:
        if os.path.exists(base_dir):
            sh_files.extend(glob.glob(f"{base_dir}/**/*", recursive=True))

    clean_sh = [
        f for f in sh_files
        if os.path.isfile(f)
        and not f.endswith((".gz", ".png", ".mo", ".pyc"))
        and os.path.getsize(f) < 100_000
    ]

    print(f"Found {len(clean_py)} Python files and {len(clean_sh)} Ubuntu CLI scripts. Merging...")

    py_count, sh_count = 0, 0
    with open(CORPUS_PATH, "w", encoding="utf-8", errors="ignore") as out_f:
        for file_path in clean_py:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                    code = in_f.read().strip()
                    if len(code) > 100 and "\x00" not in code:
                        out_f.write("<|python|>\n" + code + "\n<|endoftext|>\n")
                        py_count += 1
            except Exception:
                continue

        for file_path in clean_sh:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                    code = in_f.read().strip()
                    if len(code) > 100 and "\x00" not in code:
                        out_f.write("<|bash|>\n" + code + "\n<|endoftext|>\n")
                        sh_count += 1
            except Exception:
                continue

    size_mb = os.path.getsize(CORPUS_PATH) / (1024 * 1024)
    print(f"Merged {py_count} Python files & {sh_count} CLI scripts -> {CORPUS_PATH} ({size_mb:.2f} MB)")

def train_and_encode():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|python|>", "<|bash|>"],
        show_progress=True,
    )

    print("Training custom BPE tokenizer...")
    tokenizer.train([CORPUS_PATH], trainer)
    tokenizer_path = os.path.join(PROCESSED_DIR, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text).ids
    print(f"Total tokens encoded: {len(encoded):,}")

    # 90% Train / 10% Validation split
    n = len(encoded)
    train_ids = np.array(encoded[: int(n * 0.9)], dtype=np.uint16)
    val_ids = np.array(encoded[int(n * 0.9) :], dtype=np.uint16)

    train_ids.tofile(os.path.join(PROCESSED_DIR, "train.bin"))
    val_ids.tofile(os.path.join(PROCESSED_DIR, "val.bin"))
    print(f"Saved train.bin ({len(train_ids):,} tokens) & val.bin ({len(val_ids):,} tokens)")

if __name__ == "__main__":
    collect_corpus()
    train_and_encode()