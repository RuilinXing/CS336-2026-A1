"""Invariant tests for the parallel pre-tokenization.

These tests do NOT re-implement any pre-tokenization or BPE logic. They only
call the student's own `count_pretokens_in_parallel` with different
`num_processes` and assert the resulting Counters are identical. Rationale:

  * num_processes=1  -> the whole file is a single chunk (no splitting)
  * num_processes=N  -> the file is split into N chunks on special-token
                        boundaries

If chunking is loss-less, both must produce the exact same pre-token counts.
"""

from cs336_basics.train_bpe import count_pretokens_in_parallel
from cs336_basics.pretokenization_example import find_chunk_boundaries

SPECIAL = "<|endoftext|>"


def _write_corpus(path, num_docs=300):
    """Build a small multi-document corpus covering the PAT regex branches:
    letters, digits, punctuation, contractions, leading spaces, newlines."""
    docs = []
    for i in range(num_docs):
        docs.append(
            f"Once upon a time, child {i} had {i} apples.\n"
            f"They're happy! It costs ${i}.50 -- a steal? {i}+{i}={2 * i}"
        )
    path.write_text(SPECIAL.join(docs), encoding="utf-8")
    return path


def test_chunking_is_lossless_with_special_token(tmp_path):
    # Arrange
    corpus = _write_corpus(tmp_path / "corpus.txt")
    # Guard: ensure the corpus is actually split into >1 chunk, otherwise the
    # equality below would hold trivially without exercising the chunking.
    with open(corpus, "rb") as f:
        boundaries = find_chunk_boundaries(f, 8, SPECIAL.encode("utf-8"))
    assert len(boundaries) > 2, "corpus too small; chunking collapsed to one chunk"

    # Act
    single_chunk = count_pretokens_in_parallel(str(corpus), [SPECIAL], num_processes=1)
    many_chunks = count_pretokens_in_parallel(str(corpus), [SPECIAL], num_processes=8)

    # Assert
    assert single_chunk == many_chunks


def test_chunking_is_lossless_without_special_token(tmp_path):
    # Arrange
    corpus = _write_corpus(tmp_path / "corpus.txt")

    # Act: with no special tokens the file cannot be split, so both calls
    # degrade to a single chunk; this guards the empty-special-tokens branch.
    single_chunk = count_pretokens_in_parallel(str(corpus), [], num_processes=1)
    many_chunks = count_pretokens_in_parallel(str(corpus), [], num_processes=4)

    # Assert
    assert single_chunk == many_chunks


def test_special_token_is_not_counted_as_pretoken(tmp_path):
    # Arrange
    corpus = _write_corpus(tmp_path / "corpus.txt")

    # Act
    counts = count_pretokens_in_parallel(str(corpus), [SPECIAL], num_processes=4)

    # Assert: the special token must be stripped, never appearing as a pretoken.
    special_bytes = tuple(bytes([b]) for b in SPECIAL.encode("utf-8"))
    assert special_bytes not in counts
