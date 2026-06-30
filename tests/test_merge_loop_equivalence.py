"""Equivalence (oracle) test for the cached merge loop.

`train_bpe_merge_loop` (naive) is already pinned to the reference snapshots by
test_train_bpe, so it serves as a golden oracle here: feeding identical inputs
to the naive and the cached loop, the resulting `vocab` and `merges` must be
byte-for-byte identical.

The hand-built cases deliberately hammer the cache's trickiest branch:
repeated / overlapping pairs inside a single pretoken (e.g. "aaaa", "abab"),
where get_adjacent_pairs counts the same pair more than once.

This does NOT re-implement any BPE logic; it only compares the outputs of the
two existing loops on the same input.
"""

from collections import Counter

import pytest

from cs336_basics.train_bpe import (
    count_pretokens_in_parallel,
    init_vocab,
    train_bpe_merge_loop,
    train_bpe_merge_loop_cached,
)


def _counts(spec: dict[str, int]) -> Counter:
    """Turn {word: frequency} into a pretoken Counter of single-byte tuples,
    matching the format produced by the pre-tokenizer."""
    counts: Counter = Counter()
    for word, freq in spec.items():
        pretoken = tuple(bytes([b]) for b in word.encode("utf-8"))
        counts[pretoken] = freq
    return counts


def _run_both(counts: Counter, vocab_size: int):
    # Each loop mutates its `vocab` in place, so give each an independent copy.
    base = init_vocab([])  # 0..255 single-byte tokens
    naive = train_bpe_merge_loop(Counter(counts), dict(base), vocab_size)
    cached = train_bpe_merge_loop_cached(Counter(counts), dict(base), vocab_size)
    return naive, cached


@pytest.mark.parametrize(
    "spec",
    [
        {"aaaa": 5},                                      # overlapping (a,a)
        {"abab": 3},                                      # overlapping (a,b)
        {"aaaaaaaa": 1},                                  # long run of repeats
        {"ab": 1, "ba": 1},                               # tie -> lexicographic
        {"a": 10, "b": 5},                                # nothing to merge
        {"low": 5, "lower": 2, "newest": 6, "widest": 3},  # shared pairs
    ],
)
def test_cached_matches_naive(spec):
    # Act
    (v_naive, m_naive), (v_cached, m_cached) = _run_both(_counts(spec), vocab_size=300)

    # Assert: same merges, in the same order, and same final vocab.
    assert m_naive == m_cached
    assert v_naive == v_cached


def test_cached_matches_naive_on_text_corpus(tmp_path):
    # Arrange: a richer corpus exercised through the real pre-tokenizer.
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("They're the lowest, fastest, widest runners! " * 40)
        + "<|endoftext|>"
        + ("123 1234 12345 aaaa abab abcabc... " * 40),
        encoding="utf-8",
    )
    counts = count_pretokens_in_parallel(str(corpus), ["<|endoftext|>"], num_processes=1)

    # Act
    (v_naive, m_naive), (v_cached, m_cached) = _run_both(counts, vocab_size=320)

    # Assert
    assert m_naive == m_cached
    assert v_naive == v_cached
