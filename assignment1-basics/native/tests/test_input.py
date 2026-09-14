import pytest

import _cs336_bpe


def test_initial_vocabulary():
    vocab, merges = _cs336_bpe.train_bpe({b"\x00\x80\xff": 2**40}, 256, [])
    assert vocab == {i: bytes([i]) for i in range(256)}
    assert merges == []


def test_special_tokens_keep_order_and_are_utf8():
    vocab, merges = _cs336_bpe.train_bpe({}, 258, ["<end>", "牛", "<end>"])
    assert len(vocab) == 258
    assert vocab[256] == b"<end>"
    assert vocab[257] == "牛".encode()
    assert merges == []


@pytest.mark.parametrize(
    "counts,size,specials,error",
    [
        ({"text": 1}, 256, [], TypeError),
        ({b"a": True}, 256, [], TypeError),
        ({b"a": 1.5}, 256, [], TypeError),
        ({b"a": 0}, 256, [], ValueError),
        ({b"a": -1}, 256, [], ValueError),
        ({b"a": 2**63}, 256, [], OverflowError),
        ({b"": 1}, 256, [], ValueError),
        ({}, 255, [], ValueError),
        ({}, 256, ["<end>"], ValueError),
        ({}, 257, [""], ValueError),
        ({}, 257, [b"<end>"], TypeError),
    ],
)
def test_invalid_input(counts, size, specials, error):
    with pytest.raises(error):
        _cs336_bpe.train_bpe(counts, size, specials)


def test_train_one_merge():
    vocab, merges = _cs336_bpe.train_bpe({b"abab": 2, b"ab": 1}, 257, [])
    assert vocab[256] == b"ab"
    assert merges == [(b"a", b"b")]


def test_train_merges_non_overlapping():
    vocab, merges = _cs336_bpe.train_bpe({b"aaaa": 1}, 258, [])
    assert vocab[256] == b"aa"
    assert merges[0] == (b"a", b"a")
    assert vocab[257] == b"aaaa"
    assert merges[1] == (b"aa", b"aa")


def test_pair_count_overflow():
    with pytest.raises(OverflowError, match="Pair frequency"):
        _cs336_bpe.train_bpe({b"aaa": 2**62}, 257, [])
