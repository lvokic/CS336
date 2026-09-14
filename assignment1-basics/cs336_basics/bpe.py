"""Public assignment interfaces; BPE algorithms are intentionally unfinished."""

import os
import struct
import tempfile
from collections.abc import Iterable, Iterator
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import regex

PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

_TOKENIZER_MAGIC = b"CS336BP\0"
_TOKENIZER_VERSION = 1
_TOKENIZER_HEADER = struct.Struct("<8sIIII")
_TOKENIZER_COUNT = struct.Struct("<I")
_TOKENIZER_LENGTH = struct.Struct("<Q")


def _as_blob(value: bytes, field: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{field} must contain bytes")
    return value


def _write_blob(stream, value: bytes) -> None:
    stream.write(_TOKENIZER_LENGTH.pack(len(value)))
    stream.write(value)


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated tokenizer file")
    return value


def _read_blob(stream) -> bytes:
    size = _TOKENIZER_LENGTH.unpack(_read_exact(stream, _TOKENIZER_LENGTH.size))[0]
    return _read_exact(stream, size)


def save_tokenizer(
    path: str | os.PathLike,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> None:
    """Save a tokenizer as a compact, versioned binary file atomically."""
    special_tokens = list(special_tokens or [])
    if len(vocab) > 0xFFFFFFFF or len(merges) > 0xFFFFFFFF or len(special_tokens) > 0xFFFFFFFF:
        raise ValueError("Tokenizer is too large for this file format")
    expected_ids = set(range(len(vocab)))
    if set(vocab) != expected_ids:
        raise ValueError("vocab IDs must be contiguous and start at zero")
    if len(vocab) < 256:
        raise ValueError("vocab must contain the 256 byte tokens")
    vocab_blobs = [_as_blob(vocab[index], "vocab") for index in range(len(vocab))]
    for left, right in merges:
        _as_blob(left, "merges")
        _as_blob(right, "merges")
    if any(not isinstance(token, str) or not token for token in special_tokens):
        raise ValueError("special_tokens must contain non-empty strings")

    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    fd, temporary = tempfile.mkstemp(prefix=".cs336-tokenizer-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(
                _TOKENIZER_HEADER.pack(
                    _TOKENIZER_MAGIC,
                    _TOKENIZER_VERSION,
                    len(vocab_blobs),
                    len(merges),
                    len(special_tokens),
                )
            )
            for value in vocab_blobs:
                _write_blob(stream, value)
            for left, right in merges:
                _write_blob(stream, left)
                _write_blob(stream, right)
            for token in special_tokens:
                _write_blob(stream, token.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_tokenizer(
    path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    """Load and validate a tokenizer saved by save_tokenizer."""
    with open(path, "rb") as stream:
        magic, version, vocab_size, merge_count, special_count = _TOKENIZER_HEADER.unpack(
            _read_exact(stream, _TOKENIZER_HEADER.size)
        )
        if magic != _TOKENIZER_MAGIC or version != _TOKENIZER_VERSION:
            raise ValueError("Unsupported tokenizer file format")
        vocab = {index: _read_blob(stream) for index in range(vocab_size)}
        merges = [(_read_blob(stream), _read_blob(stream)) for _ in range(merge_count)]
        special_tokens = [_read_blob(stream).decode("utf-8") for _ in range(special_count)]
        if stream.read(1):
            raise ValueError("Unexpected trailing data in tokenizer file")
    if len(vocab) < 256 or any(len(vocab[index]) == 0 for index in range(256)):
        raise ValueError("Invalid tokenizer vocabulary")
    return vocab, merges, special_tokens


def _native():
    try:
        import _cs336_bpe
    except ModuleNotFoundError as exc:
        if exc.name != "_cs336_bpe":
            raise
        raise ImportError(
            "Build the C++ extension from the assignment root with: "
            "uv pip install --python .venv/bin/python ./native"
        ) from exc
    return _cs336_bpe


def _count_pretokens_chunk(text: str, special_tokens: tuple[str, ...]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    if special_tokens:
        pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        segments = regex.split(pattern, text)
    else:
        segments = [text]
    for segment in segments:
        for match in PRETOKEN_PATTERN.finditer(segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _iter_file_chunks(
    path: str | os.PathLike,
    chunk_bytes: int,
    special_tokens: tuple[str, ...] = (),
) -> Iterator[str]:
    """Yield chunks at boundaries that cannot change pre-tokenization.

    A bare newline is not always a safe boundary: the next chunk may begin
    with spaces, and ``PRETOKEN_PATTERN`` can then match that space together
    with the following word.  We therefore keep the complete ASCII
    whitespace run with the preceding chunk.  We also avoid cutting through a
    special token, including special tokens that contain a newline.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    special_bytes = tuple(token.encode("utf-8") for token in special_tokens)
    pending = b""
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            pending += block
            newline = pending.rfind(b"\n")
            if newline < 0:
                continue

            # Keep the complete whitespace run on the right side.  The regex
            # treats trailing whitespace differently when it can see the
            # following non-whitespace text, so ending a chunk in whitespace
            # would not be semantics-preserving.
            cut = newline
            while cut > 0 and pending[cut - 1] in b" \t\r\n\v\f":
                cut -= 1
            if cut == 0:
                continue

            # A special token containing the candidate boundary must remain
            # in one piece.  If necessary, retreat to an earlier newline.
            while True:
                crossing_start = None
                for token in special_bytes:
                    start = pending.rfind(token, 0, cut)
                    if start >= 0 and start + len(token) > cut:
                        crossing_start = start if crossing_start is None else min(crossing_start, start)
                if crossing_start is None:
                    break
                previous_newline = pending.rfind(b"\n", 0, crossing_start)
                if previous_newline < 0:
                    cut = 0
                    break
                cut = previous_newline
                while cut > 0 and pending[cut - 1] in b" \t\r\n\v\f":
                    cut -= 1

            if cut == 0:
                continue
            chunk, pending = pending[:cut], pending[cut:]
            yield chunk.decode("utf-8")
    if pending:
        yield pending.decode("utf-8")


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int,
    chunk_bytes: int,
) -> Counter[bytes]:
    file_size = os.path.getsize(input_path)
    if num_workers <= 1 or file_size <= chunk_bytes:
        with open(input_path, encoding="utf-8", newline="") as stream:
            return _count_pretokens_chunk(stream.read(), tuple(special_tokens))

    special_tuple = tuple(special_tokens)
    chunks = _iter_file_chunks(input_path, chunk_bytes, special_tuple)
    counts: Counter[bytes] = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for local_counts in executor.map(
            _count_pretokens_chunk,
            chunks,
            repeat(special_tuple),
            chunksize=1,
        ):
            counts.update(local_counts)
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """TODO: prepare the corpus, then call the native training interface."""
    unknown = set(kwargs) - {"num_workers", "chunk_bytes"}
    if unknown:
        raise TypeError(f"Unsupported options: {', '.join(sorted(unknown))}")
    num_workers = kwargs.get("num_workers", max(1, min(os.cpu_count() or 1, 32)))
    chunk_bytes = kwargs.get("chunk_bytes", 64 * 1024 * 1024)
    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    if not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must accommodate 256 bytes and all special tokens")
    native = _native()

    pretoken_counts = _count_pretokens(input_path, special_tokens, num_workers, chunk_bytes)

    return native.train_bpe(dict(pretoken_counts), vocab_size, special_tokens)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self._bytes_to_id = {value: key for key, value in self.vocab.items()}
        self._special_ids = {
            token: self._bytes_to_id[token.encode()] for token in self.special_tokens
        }
        self._special_pattern = (
            regex.compile(
                "|".join(regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            )
            if self.special_tokens
            else None
        )
        self._core = _native().TokenizerCore(self.vocab, self.merges, self.special_tokens)

    def save(self, path: str | os.PathLike) -> None:
        save_tokenizer(path, self.vocab, self.merges, self.special_tokens)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Tokenizer":
        vocab, merges, special_tokens = load_tokenizer(path)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text after preserving special tokens and pre-tokenizing it."""
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return self._encode_all(text)

    def _encode_all(self, text: str) -> list[int]:
        if not text:
            return []
        result: list[int] = []
        cursor = 0
        if self._special_pattern is None:
            segments = [(text, None)]
        else:
            segments = []
            for match in self._special_pattern.finditer(text):
                segments.append((text[cursor : match.start()], None))
                segments.append((match.group(), match.group()))
                cursor = match.end()
            segments.append((text[cursor:], None))
        for segment, special in segments:
            if special is not None:
                result.append(self._special_ids[special])
            else:
                for match in PRETOKEN_PATTERN.finditer(segment):
                    result.extend(self._core.encode_pretoken(match.group().encode("utf-8")))
        return result

    def _encode_normal_prefix(self, text: str) -> tuple[list[int], str]:
        """Encode stable pretokens and return the possibly incomplete suffix."""
        if not text:
            return [], ""
        matches = PRETOKEN_PATTERN.finditer(text)
        previous = None
        result: list[int] = []
        for match in matches:
            if previous is not None:
                result.extend(self._core.encode_pretoken(previous.group().encode("utf-8")))
            previous = match
        if previous is None:
            return [], text
        if previous.end() == len(text):
            return result, text[previous.start() :]
        result.extend(self._core.encode_pretoken(previous.group().encode("utf-8")))
        return result, text[previous.end() :]

    def _partial_special_start(self, text: str) -> int | None:
        if self._special_pattern is None:
            return None
        best: int | None = None
        for token in self.special_tokens:
            maximum = min(len(token) - 1, len(text))
            for size in range(maximum, 0, -1):
                if text.endswith(token[:size]):
                    start = len(text) - size
                    best = start if best is None else min(best, start)
                    break
        return best

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into a Unicode string."""
        return self._core.decode_bytes(ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks while preserving tokens across chunk boundaries."""
        pending = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("iterable must yield str values")
            pending += chunk
            partial_start = self._partial_special_start(pending)
            if partial_start is not None:
                stable = pending[:partial_start]
                for token_id in self._encode_all(stable):
                    yield token_id
                pending = pending[partial_start:]
                continue

            last_special = None
            if self._special_pattern is not None:
                for match in self._special_pattern.finditer(pending):
                    last_special = match
            if last_special is not None:
                stable = pending[: last_special.end()]
                for token_id in self._encode_all(stable):
                    yield token_id
                pending = pending[last_special.end() :]
                continue

            encoded, pending = self._encode_normal_prefix(pending)
            yield from encoded

        yield from self._encode_all(pending)
