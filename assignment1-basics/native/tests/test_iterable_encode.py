from cs336_basics.bpe import Tokenizer


def make_tokenizer():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"hello"
    vocab[257] = b"ab"
    vocab[258] = b"<|endoftext|>"
    return Tokenizer(vocab, [(b"a", b"b")], ["<|endoftext|>"])


def test_iterable_preserves_pretoken_across_chunks():
    tokenizer = make_tokenizer()
    chunks = ["hel", "lo a", "b wor", "ld"]
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode("hello ab world")


def test_iterable_preserves_split_special_token():
    tokenizer = make_tokenizer()
    chunks = ["hello <|end", "oftext|> world"]
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode("hello <|endoftext|> world")


def test_iterable_does_not_accumulate_input():
    tokenizer = make_tokenizer()
    chunks = ("a" for _ in range(1000))
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode("a" * 1000)
