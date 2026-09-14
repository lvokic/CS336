import pytest

import _cs336_bpe


def make_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"hello"
    vocab[257] = "牛".encode()
    vocab[258] = b"\x00\xff"
    return vocab


def test_decode_concatenates_arbitrary_token_bytes():
    core = _cs336_bpe.TokenizerCore(make_vocab(), [], [])
    assert core.decode_bytes([256, 32, 257, 258]) == "hello 牛".encode() + b"\x00\xff"
    assert core.decode_bytes([]) == b""


@pytest.mark.parametrize("ids,error", [([-1], IndexError), ([999], IndexError), ([True], TypeError), ([1.5], TypeError)])
def test_decode_rejects_invalid_ids(ids, error):
    core = _cs336_bpe.TokenizerCore(make_vocab(), [], [])
    with pytest.raises(error):
        core.decode_bytes(ids)
