# TinyStories experiment workflow

The tokenizer file and the tokenized corpus are different artifacts:

- `data/tinystories_tokenizer.bin` stores BPE vocabulary and merge rules.
- `*_tokens.bin` stores a flat little-endian `uint16` array for fast training.

## 1. Encode train and validation data

Run once for each text split. The encoder streams both input and output, so it
does not require the full corpus in RAM.

```bash
uv run --no-sync python experiments/prepare_tokens.py \
  --tokenizer data/tinystories_tokenizer.bin \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train_tokens.bin

uv run --no-sync python experiments/prepare_tokens.py \
  --tokenizer data/tinystories_tokenizer.bin \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories_valid_tokens.bin
```

Each output has a sibling JSON metadata file. Check that the reported maximum
token ID is below 10,000 before training.

## 2. Overfit one minibatch

Before a long run, temporarily make `get_batch` deterministic or write a short
notebook that retains one sampled `(inputs, targets)` pair and repeatedly calls
`train_step`. Its loss should approach zero. Do not proceed if it cannot.

## 3. Smoke test the end-to-end pipeline

```bash
uv run --no-sync python experiments/train_tinystories.py \
  --train-tokens data/tinystories_train_tokens.bin \
  --valid-tokens data/tinystories_valid_tokens.bin \
  --output-dir experiments/tinystories_smoke \
  --steps 100 --batch-size 4 --eval-interval 50 --eval-batches 2
```

Confirm that `config.json`, `metrics.json`, and `latest.pt` appear. Then check
that `valid_loss` is finite and that a resumed run works.

## 4. Baseline run

The default model is the TinyStories handout configuration: 10,000 vocabulary,
256 context length, 512 model width, 4 layers, 16 heads, and SwiGLU width 1344.

```bash
uv run --no-sync python experiments/train_tinystories.py \
  --train-tokens data/tinystories_train_tokens.bin \
  --valid-tokens data/tinystories_valid_tokens.bin \
  --output-dir experiments/tinystories_baseline \
  --steps 40000 --batch-size 32
```

This processes `40,000 * 32 * 256 = 327,680,000` training tokens. If the GPU
fits a larger batch, preserve the same token budget by decreasing steps:
`steps = target_tokens / (batch_size * context_length)`.

## 5. Resume safely

```bash
uv run --no-sync python experiments/train_tinystories.py \
  --train-tokens data/tinystories_train_tokens.bin \
  --valid-tokens data/tinystories_valid_tokens.bin \
  --output-dir experiments/tinystories_baseline \
  --steps 40000 --batch-size 32 \
  --resume experiments/tinystories_baseline/latest.pt
```

`--steps` is always the final global step. The optimizer state and cosine
schedule continue from the checkpoint rather than restarting warmup.

## 6. Evaluate experiments

The target TinyStories validation loss is about 1.45 or lower. Compare runs by
validation loss, perplexity (`exp(validation_loss)`), tokens/sec, and samples
generated from their checkpoints. Change one variable at a time: learning rate,
warmup, weight decay, batch size, or total token budget.

## 7. Decode generated text

Implement `cs336_basics/generation.py` before using the generation launcher.
Then generate from a trained checkpoint:

```bash
uv run --no-sync python experiments/generate_tinystories.py \
  --tokenizer data/tinystories_tokenizer.bin \
  --checkpoint experiments/tinystories_baseline/latest.pt \
  --config experiments/tinystories_baseline/config.json \
  --prompt "Once upon a time" \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.9
```
