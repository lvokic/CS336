# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Train and save tokenizers

Train the TinyStories tokenizer:

```sh
uv run --no-sync python - <<'PY'
from cs336_basics.bpe import train_bpe, save_tokenizer

vocab, merges = train_bpe(
    "data/TinyStoriesV2-GPT4-train.txt",
    10_000,
    ["<|endoftext|>"],
    num_workers=16,
    chunk_bytes=64 * 1024 * 1024,
)
save_tokenizer(
    "data/tinystories_tokenizer.bin",
    vocab,
    merges,
    ["<|endoftext|>"],
)

print(len(vocab), len(merges))
print("saved")
PY
```

Load and verify the TinyStories tokenizer:

```sh
uv run --no-sync python - <<'PY'
from cs336_basics.bpe import Tokenizer

tokenizer = Tokenizer.load("data/tinystories_tokenizer.bin")

text = "Once upon a time."
ids = tokenizer.encode(text)

print("ids:", ids)
print("decoded:", tokenizer.decode(ids))
PY
```

Train and save the OpenWebText tokenizer. The standard configuration uses a
32,000-token vocabulary:

```sh
time uv run --no-sync python - <<'PY'
from cs336_basics.bpe import train_bpe, save_tokenizer

vocab, merges = train_bpe(
    "data/owt_train.txt",
    32_000,
    ["<|endoftext|>"],
    num_workers=32,
    chunk_bytes=64 * 1024 * 1024,
)

save_tokenizer(
    "data/owt_tokenizer.bin",
    vocab,
    merges,
    ["<|endoftext|>"],
)

print(f"vocab size: {len(vocab)}")
print(f"merge count: {len(merges)}")
print("saved to data/owt_tokenizer.bin")
PY
```

Load and verify the OpenWebText tokenizer:

```sh
uv run --no-sync python - <<'PY'
from cs336_basics.bpe import Tokenizer, load_tokenizer

vocab, merges, specials = load_tokenizer("data/owt_tokenizer.bin")
token_id, token = max(vocab.items(), key=lambda item: len(item[1]))

print("token id:", token_id)
print("byte length:", len(token))
print("token:", repr(token))
PY
```

### Compute tokenizer compression ratio

The compression ratio is measured as UTF-8 bytes per token:

```text
compression ratio = UTF-8 byte count / token count
```

The following command computes the ratio for both tokenizers on ten sampled
TinyStories and OpenWebText documents:

```sh
uv run --no-sync python - <<'PY'
from cs336_basics.bpe import Tokenizer

SPECIAL = "<|endoftext|>"


def sample_documents(path, count=10):
    documents = []
    pending = ""

    with open(path, encoding="utf-8") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), ""):
            pending += chunk
            parts = pending.split(SPECIAL)

            for part in parts[:-1]:
                if part:
                    documents.append(part)
                    if len(documents) == count:
                        return documents

            pending = parts[-1]

    if pending and len(documents) < count:
        documents.append(pending)

    return documents[:count]


def compression_ratio(tokenizer, documents):
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))

    return total_bytes / total_tokens


tiny_tokenizer = Tokenizer.load("data/tinystories_tokenizer.bin")
owt_tokenizer = Tokenizer.load("data/owt_tokenizer.bin")

tiny_docs = sample_documents("data/TinyStoriesV2-GPT4-train.txt")
owt_docs = sample_documents("data/owt_train.txt")

for name, tokenizer in [
    ("TinyStories tokenizer", tiny_tokenizer),
    ("OpenWebText tokenizer", owt_tokenizer),
]:
    print(f"\n{name}")
    print("TinyStories:", compression_ratio(tokenizer, tiny_docs), "bytes/token")
    print("OpenWebText:", compression_ratio(tokenizer, owt_docs), "bytes/token")
PY
```
