import struct

import pytest

from cs336_basics.bpe import load_tokenizer, save_tokenizer


def sample_tokenizer():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"aba"})
    merges = [(b"a", b"b"), (b"ab", b"a")]
    return vocab, merges, ["<|endoftext|>", "牛"]


def test_binary_tokenizer_roundtrip(tmp_path):
    expected = sample_tokenizer()
    path = tmp_path / "tokenizer.bin"
    save_tokenizer(path, *expected)
    assert load_tokenizer(path) == expected
    assert path.read_bytes()[:8] == b"CS336BP\0"


def test_save_is_atomic_on_invalid_input(tmp_path):
    path = tmp_path / "tokenizer.bin"
    path.write_bytes(b"old")
    vocab, merges, specials = sample_tokenizer()
    vocab[999] = b"bad"
    with pytest.raises(ValueError):
        save_tokenizer(path, vocab, merges, specials)
    assert path.read_bytes() == b"old"


def test_rejects_corrupt_or_trailing_file(tmp_path):
    path = tmp_path / "tokenizer.bin"
    vocab, merges, specials = sample_tokenizer()
    save_tokenizer(path, vocab, merges, specials)
    path.write_bytes(path.read_bytes() + b"junk")
    with pytest.raises(ValueError, match="trailing"):
        load_tokenizer(path)


def test_rejects_bad_magic(tmp_path):
    path = tmp_path / "tokenizer.bin"
    path.write_bytes(struct.pack("<8sIIII", b"bad\0\0\0\0\0", 1, 0, 0, 0))
    with pytest.raises(ValueError, match="Unsupported"):
        load_tokenizer(path)
