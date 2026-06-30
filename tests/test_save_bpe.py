"""TDD harness for saving the trained BPE vocab/merges to disk.

This will FAIL until you implement `cs336_basics.train_bpe.save_bpe`. It does
NOT implement saving for you. It only:
  1. calls your `save_bpe`, then
  2. reads the files back with the SAME GPT-2 byte decoder used by the course's
     get_tokenizer_from_vocab_merges_path (see tests/test_tokenizer.py),
  3. and asserts the vocab/merges round-trip exactly.

Because step 2 uses the course decoder, a green test also means your files are
loadable by the Tokenizer.from_files path you'll write next.

Assumed interface (rename in the import/calls if you choose differently):
    save_bpe(vocab, merges, vocab_path, merges_path) -> None
      * vocab.json : {token_string: id}, GPT-2 byte->unicode encoded, utf-8
      * merges.txt : one "enc(tok1) enc(tok2)" per line, in merge order
"""

import json

from cs336_basics.train_bpe import init_vocab, save_bpe

from .common import gpt2_bytes_to_unicode


def _make_vocab_and_merges():
    # special token + all 256 single bytes (covers unprintable / high bytes
    # like 0x00 and 0xFF that would break a naive bytes.decode("utf-8"))
    vocab = init_vocab(["<|endoftext|>"])
    # a few "merged" tokens, as a real training run would append them; note the
    # space byte b" " (encodes to 'Ġ') and a multi-byte token b"the"
    merges = [(b"t", b"h"), (b"th", b"e"), (b" ", b"the")]
    next_id = len(vocab)
    for a, b in merges:
        vocab[next_id] = a + b
        next_id += 1
    return vocab, merges


def _load_with_gpt2_decoder(vocab_path, merges_path):
    byte_decoder = {ch: b for b, ch in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {
        idx: bytes(byte_decoder[ch] for ch in token)
        for token, idx in raw_vocab.items()
    }
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and len(line.split(" ")) == 2:
                a, b = line.split(" ")
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in a),
                        bytes(byte_decoder[ch] for ch in b),
                    )
                )
    return vocab, merges


def test_save_bpe_roundtrips_via_gpt2_decoder(tmp_path):
    # Arrange
    vocab, merges = _make_vocab_and_merges()
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    # Act
    save_bpe(vocab, merges, vocab_path, merges_path)
    loaded_vocab, loaded_merges = _load_with_gpt2_decoder(vocab_path, merges_path)

    # Assert: exact recovery, including unprintable/high bytes, the space byte,
    # and the merge order.
    assert loaded_vocab == vocab
    assert loaded_merges == merges


def test_save_bpe_vocab_json_shape(tmp_path):
    # Arrange
    vocab, merges = _make_vocab_and_merges()
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    # Act
    save_bpe(vocab, merges, vocab_path, merges_path)
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Assert: stored as {token_string: int_id}, not the other way around, and
    # the special token (all-printable-ASCII bytes) is kept verbatim at id 0.
    assert all(isinstance(k, str) for k in raw)
    assert all(isinstance(v, int) for v in raw.values())
    assert raw["<|endoftext|>"] == 0
