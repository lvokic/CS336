from pathlib import Path

from cs336_basics.bpe import _count_pretokens, train_bpe


def test_parallel_pretoken_counts_match_serial(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("alpha beta\nalpha <|endoftext|> gamma\n" * 100, encoding="utf-8")
    specials = ["<|endoftext|>"]
    serial = _count_pretokens(path, specials, num_workers=1, chunk_bytes=32)
    parallel = _count_pretokens(path, specials, num_workers=2, chunk_bytes=32)
    assert parallel == serial


def test_parallel_pretoken_keeps_whitespace_and_special_boundaries(tmp_path: Path):
    path = tmp_path / "boundary.txt"
    path.write_text(
        "alpha\n   beta\n<|endoftext|>\n gamma\n", encoding="utf-8"
    )
    serial = _count_pretokens(path, ["<|endoftext|>"], num_workers=1, chunk_bytes=7)
    parallel = _count_pretokens(path, ["<|endoftext|>"], num_workers=2, chunk_bytes=7)
    assert parallel == serial


def test_parallel_training_matches_serial(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text("abab ab\n" * 100, encoding="utf-8")
    serial = train_bpe(path, 270, [], num_workers=1, chunk_bytes=8)
    parallel = train_bpe(path, 270, [], num_workers=2, chunk_bytes=8)
    assert parallel == serial
