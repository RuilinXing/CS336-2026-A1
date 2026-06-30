import json
import os
import regex as re
from collections.abc import Iterable, Iterator

from .train_bpe import PAT, gpt2_bytes_to_unicode

Pair = tuple[bytes, bytes]

# Replacement bytes for any id missing from the vocab (U+FFFD).
_REPLACEMENT = "�".encode("utf-8")


class Tokenizer:
    """Byte-level BPE tokenizer.

    Encodes text to token ids and decodes ids back to text using a fixed
    vocabulary (id -> bytes) and an ordered list of BPE merges.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse lookup: token bytes -> id.
        self.token_to_id: dict[bytes, int] = {
            token: token_id for token_id, token in vocab.items()
        }
        # Merge priority: lower rank merges first.
        self.merge_ranks: dict[Pair, int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        self._pat = re.compile(PAT)
        # Sort special tokens by length (desc) so the longest match wins when
        # special tokens overlap (e.g. "<|eot|><|eot|>" before "<|eot|>").
        self._special_set = set(self.special_tokens)
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Build a Tokenizer from a GPT-2-style vocab.json and merges.txt.

        This matches the format produced by `train_bpe.save_bpe`:
            * vocab.json : {token_string: id}, GPT-2 byte->unicode encoded
            * merges.txt : one "enc(tok1) enc(tok2)" per line, in merge order
        """
        byte_decoder = {ch: b for b, ch in gpt2_bytes_to_unicode().items()}

        with open(vocab_path, encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        vocab: dict[int, bytes] = {
            token_id: bytes(byte_decoder[ch] for ch in token_str)
            for token_str, token_id in raw_vocab.items()
        }

        merges: list[Pair] = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or len(line.split(" ")) != 2:
                    continue
                left, right = line.split(" ")
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in left),
                        bytes(byte_decoder[ch] for ch in right),
                    )
                )

        # Append any special tokens that are not already in the vocab.
        if special_tokens:
            existing = set(vocab.values())
            for tok in special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in existing:
                    vocab[len(vocab)] = tok_bytes
                    existing.add(tok_bytes)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self._encode_to_ids(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Stream line-by-line so we never hold the whole corpus in memory.
        # For GPT-2 pre-tokenization, per-line encoding matches whole-text
        # encoding because the regex isolates trailing whitespace at line ends.
        for line in iterable:
            yield from self._encode_to_ids(line)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab.get(token_id, _REPLACEMENT) for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _encode_to_ids(self, text: str) -> Iterator[int]:
        for segment in self._split_on_special(text):
            if not segment:
                continue
            if segment in self._special_set:
                yield self.token_to_id[segment.encode("utf-8")]
                continue
            for match in self._pat.finditer(segment):
                yield from self._bpe_encode(match.group().encode("utf-8"))

    def _split_on_special(self, text: str) -> list[str]:
        if self._special_pattern is None:
            return [text]
        # Capturing group keeps the special tokens as their own segments.
        return self._special_pattern.split(text)

    def _bpe_encode(self, token_bytes: bytes) -> list[int]:
        parts: list[bytes] = [bytes([b]) for b in token_bytes]

        while len(parts) >= 2:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            parts = self._apply_merge(parts, best_pair)

        return [self.token_to_id[part] for part in parts]

    @staticmethod
    def _apply_merge(parts: list[bytes], pair: Pair) -> list[bytes]:
        merged: list[bytes] = []
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                merged.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        return merged
