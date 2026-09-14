import pytest

import _cs336_bpe


def vocab_with_merges():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"aba", 258: b"abab"})
    return vocab, [(b"a", b"b"), (b"ab", b"a"), (b"aba", b"b")]


def test_encode_uses_merge_order():
    vocab, merges = vocab_with_merges()
    core = _cs336_bpe.TokenizerCore(vocab, merges, [])
    assert core.encode_pretoken(b"abab") == [256, 256]
    assert core.encode_pretoken(b"ababa") == [256, 257]
    assert core.encode_pretoken("牛".encode()) == list("牛".encode())


def test_encode_handles_overlapping_pair_candidates():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"aa"
    vocab[257] = b"aaa"
    core = _cs336_bpe.TokenizerCore(vocab, [(b"a", b"a"), (b"aa", b"a")], [])
    assert core.encode_pretoken(b"aaaa") == [256, 256]


def test_encode_rejects_bad_merges():
    vocab = {i: bytes([i]) for i in range(256)}
    with pytest.raises(ValueError):
        _cs336_bpe.TokenizerCore(vocab, [(b"a", b"b")], [])
