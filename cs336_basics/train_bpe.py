import os
import json
import regex as re
from pathlib import Path
from functools import lru_cache
from collections import Counter
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries

# corpus path (anchored to repo root so it works regardless of CWD)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OWT_TRAIN_PATH = os.path.join(_DATA_DIR, 'owt_train.txt') # 11.92 GB
OWT_VALID_PATH = os.path.join(_DATA_DIR, 'owt_valid.txt') # 290 MB
TINYSTORIES_TRAIN_PATH = os.path.join(_DATA_DIR, 'TinyStoriesV2-GPT4-train.txt') # 2.23 GB
TINYSTORIES_VALID_PATH = os.path.join(_DATA_DIR, 'TinyStoriesV2-GPT4-valid.txt') # 22.5 MB

# end of document token 
END_OF_TEXT_TOKEN = "<|endoftext|>"

# GPT-2 预处理正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pretoken = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

########
######## Define pre-tokenization 
########
"""
打开 corpus 文件
→ 找 chunk boundaries
→ 根据 start/end 读取每个 chunk
→ 将 byte decode 成 string
→ 对每个 chunk 做 pre-tokenization
→ 统计 pre-token counts
"""


def count_pretokens_in_chunk(
    args: tuple[str, int, int, list[str]],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args

    local_counts: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    for document in split_by_special_tokens(chunk_text, special_tokens):
        for match in re.finditer(PAT, document):
            pretoken = match.group()
            pretoken_bytes = tuple(
                bytes([b]) for b in pretoken.encode("utf-8")
            )
            local_counts[pretoken_bytes] += 1

    return local_counts


# parallelizing pre-tokenization 
def count_pretokens_in_parallel(
    input_path: str, 
    special_tokens: list[str],
    num_processes: int = 4
)-> Counter[tuple[bytes, ...]]:     
    with open(input_path, "rb") as f:
        if not special_tokens: 
            f.seek(0, os.SEEK_END)
            file_size = f.tell() 
            f.seek(0) 
            
            boundaries = [0, file_size]
        else: 
            split_special_token = special_tokens[0].encode('utf-8')
            
            # 将文件拆分成 num_processes 个 boundaries
            # 如 num_processes = 4, 将文件切分为4块独立连续的部分 
            # e.g [0, 251203, 502991, 749882, 1000000]
            boundaries = find_chunk_boundaries(
                f, 
                num_processes, 
                split_special_token
            )
    
    # workers 的 tasks pool, 每个task 为一个 chunk
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    total_counts = Counter()

    # 分配给每个work 的任务
    with Pool(processes=num_processes) as pool:
        partial_counts_list = pool.map(count_pretokens_in_chunk, tasks)

    for partial_counts in partial_counts_list:
        total_counts.update(partial_counts)

    return total_counts
    
      
def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    return re.split(pattern, text)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]: 
    vocab: dict[int, bytes] = {} 
    for token in special_tokens: 
        vocab[len(vocab)] = token.encode('utf-8')
    for i in range(256): 
        vocab[len(vocab)] = bytes([i])
    
    return vocab


def merge_pretoken(
    pretoken: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged = []
    i = 0

    while i < len(pretoken):
        if (
            i < len(pretoken) - 1
            and pretoken[i] == pair_to_merge[0]
            and pretoken[i + 1] == pair_to_merge[1]
        ):
            merged.append(pretoken[i] + pretoken[i + 1])
            i += 2
        else:
            merged.append(pretoken[i])
            i += 1

    return tuple(merged)


def get_adjacent_pairs(pretoken: Pretoken) -> list[Pair]:
    return [
        (pretoken[i], pretoken[i + 1])
        for i in range(len(pretoken) - 1)
    ]


def initialize_pair_cache(
    pretoken_counts: Counter[Pretoken],
) -> tuple[Counter[Pair], dict[Pair, set[Pretoken]]]:
    pair_counts: Counter[Pair] = Counter()
    pair_to_pretokens: dict[Pair, set[Pretoken]] = {}

    for pretoken, count in pretoken_counts.items():
        for pair in get_adjacent_pairs(pretoken):
            pair_counts[pair] += count
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    return pair_counts, pair_to_pretokens


def remove_pretoken_from_cache(
    pretoken: Pretoken,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_pretokens: dict[Pair, set[Pretoken]],
) -> None:
    for pair in get_adjacent_pairs(pretoken):
        pair_counts[pair] -= count

        if pair in pair_to_pretokens:
            pair_to_pretokens[pair].discard(pretoken)

        if pair_counts[pair] <= 0:
            del pair_counts[pair]
            pair_to_pretokens.pop(pair, None)


def add_pretoken_to_cache(
    pretoken: Pretoken,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_pretokens: dict[Pair, set[Pretoken]],
) -> None:
    for pair in get_adjacent_pairs(pretoken):
        pair_counts[pair] += count
        pair_to_pretokens.setdefault(pair, set()).add(pretoken)


def train_bpe_merge_loop_cached(
    pretoken_counts: Counter[Pretoken],
    vocab: dict[int, bytes],
    final_vocab_size: int,
    verbose: bool = False,
    log_every: int = 100,
) -> tuple[dict[int, bytes], list[Pair]]:
    import time
    import psutil

    merges: list[Pair] = []
    target_merges = final_vocab_size - len(vocab)
    start_time = time.perf_counter()
    process = psutil.Process()

    # 避免直接修改外部传进来的 Counter
    pretoken_counts = Counter(pretoken_counts)

    # 一开始只全量统计一次
    pair_counts, pair_to_pretokens = initialize_pair_cache(pretoken_counts)

    while len(vocab) < final_vocab_size:
        if not pair_counts:
            break

        # 先按 frequency 最大选；频率相同选 lexicographically greater pair
        best_pair, _ = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )

        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # 只更新包含 best_pair 的 pre-token
        affected_pretokens = list(pair_to_pretokens.get(best_pair, set()))

        for old_pretoken in affected_pretokens:
            count = pretoken_counts.pop(old_pretoken, 0)

            if count == 0:
                continue

            # 1. 删除 old_pretoken 对 pair_counts / index 的贡献
            remove_pretoken_from_cache(
                pretoken=old_pretoken,
                count=count,
                pair_counts=pair_counts,
                pair_to_pretokens=pair_to_pretokens,
            )

            # 2. merge old_pretoken
            new_pretoken = merge_pretoken(
                pretoken=old_pretoken,
                pair_to_merge=best_pair,
            )

            # 3. 加入新的 pre-token count
            pretoken_counts[new_pretoken] += count

            # 4. 加入 new_pretoken 对 pair_counts / index 的贡献
            add_pretoken_to_cache(
                pretoken=new_pretoken,
                count=count,
                pair_counts=pair_counts,
                pair_to_pretokens=pair_to_pretokens,
            )

        # Progress log every `log_every` merges.
        if verbose and len(merges) % log_every == 0:
            elapsed = time.perf_counter() - start_time
            rss_gb = process.memory_info().rss / 1024**3
            print(
                f"[merge {len(merges)}/{target_merges}] "
                f"vocab={len(vocab)} "
                f"rss={rss_gb:.2f}GB "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = count_pretokens_in_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )

    vocab = init_vocab(
        special_tokens=special_tokens,
    )

    vocab, merges = train_bpe_merge_loop_cached(
        pretoken_counts=pretoken_counts,
        vocab=vocab,
        final_vocab_size=vocab_size,
        verbose=verbose,
    )

    return vocab, merges


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 byte-to-unicode mapping.

    It maps each byte value 0..255 to a Unicode string character so that
    arbitrary bytes can be represented safely in a JSON vocabulary.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {b: chr(c) for b, c in zip(bs, cs)}


def encode_token_bytes_for_gpt2_json(token: bytes) -> str:
    """
    Convert a bytes token into GPT-2's reversible unicode string form.
    """
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token)


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """
    Save BPE vocab and merges in GPT-2-style format.

    vocab.json format:
        {token_str: token_id}

    merges.txt format:
        one merge per line, in merge order:
        token1 token2
    """
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.parent.mkdir(parents=True, exist_ok=True)

    # Important: vocab JSON direction is {token_str: id}, not {id: token_str}
    encoded_vocab: dict[str, int] = {
        encode_token_bytes_for_gpt2_json(token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(encoded_vocab, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = encode_token_bytes_for_gpt2_json(left)
            right_str = encode_token_bytes_for_gpt2_json(right)
            f.write(f"{left_str} {right_str}\n")


if __name__ == "__main__":
    import time

    INPUT_PATH = OWT_TRAIN_PATH
    VOCAB_SIZE = 32_000
    special_tokens = [END_OF_TEXT_TOKEN]
    num_processes = 8

    vocab_out = os.path.join(_DATA_DIR, "owt_vocab.json")
    merges_out = os.path.join(_DATA_DIR, "owt_merges.txt")

    print(f"Training BPE on {INPUT_PATH}")
    print(f"  vocab_size={VOCAB_SIZE}, special_tokens={special_tokens}, num_processes={num_processes}")

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        num_processes=num_processes,
        verbose=True,
    )
    elapsed = time.perf_counter() - start

    save_bpe(vocab, merges, vocab_out, merges_out)

    longest_token = max(vocab.values(), key=len)
    print(f"Done in {elapsed:.1f}s")
    print(f"  vocab size: {len(vocab)}  |  merges: {len(merges)}")
    print(f"  longest token: {longest_token!r}  ({len(longest_token)} bytes)")
    print(f"  saved vocab  -> {vocab_out}")
    print(f"  saved merges -> {merges_out}")
