import os
import regex as re 
from collections import Counter

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

def count_pretokens(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_processes: int = 4
)-> Counter[tuple[bytes, ...]]: 
    pretoken_counts = Counter()
    
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
    
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start) # 表示移动到这个 chunk 的开始位置
            # 1. 读取 chunk 的 byte 内容
            chunk_bytes = f.read(end - start)
            # 2. decode from bytes to string
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")  
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            for document in split_by_special_tokens(chunk_text, special_tokens): 
                for match in re.finditer(PAT, document): 
                    pretoken = match.group() # 得到匹配pretoken的string表示形式
                    # 将pretoken string形式编码成 utf-8  
                    pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                    pretoken_counts[pretoken_bytes] += 1
            
    return pretoken_counts
               
def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    return re.split(pattern, text)


#################
###### vocabulary initlization + train BPE 
###############
# 

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]: 
    vocab: dict[int, bytes] = {} 
    for token in special_tokens: 
        vocab[len(vocab)] = token.encode('utf-8')
    for i in range(256): 
        vocab[len(vocab)] = bytes([i])
    
    return vocab


### merge loop 
""" 
1. 统计所有相邻 token pair 的频率
2. 找到频率最高的 pair
3. 把这个 pair 合并成一个新 token
4. 更新 pretoken_counts 和 vocab
"""

def compute_pair_counts(
    pretoken_counts: Counter[tuple[bytes, ...]]
) -> Counter[tuple[bytes, bytes]]: 
    pair_counts = Counter() 
    
    for pretoken, count in pretoken_counts.items(): 
        for i in range(len(pretoken) - 1): 
            pair = (pretoken[i], pretoken[i+1])
            pair_counts[pair] += count 
    
    return pair_counts


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


def merge_pair_in_counts(
    pretoken_counts: Counter[tuple[bytes, ...]],
    pair_to_merge: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    new_counts = Counter()

    for pretoken, count in pretoken_counts.items():
        new_pretoken = merge_pretoken(pretoken, pair_to_merge)
        new_counts[new_pretoken] += count

    return new_counts


def train_bpe_merge_loop(
    pretoken_counts: Counter[tuple[bytes, ...]],
    vocab: dict[int, bytes],
    final_vocab_size: int 
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < final_vocab_size:
        pair_counts = compute_pair_counts(pretoken_counts)

        # 没有任何 pair 可以继续 merge
        if not pair_counts:
            break

        best_pair, _ = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )

        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        pretoken_counts = merge_pair_in_counts(
            pretoken_counts,
            best_pair,
        )

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = count_pretokens(
        input_path=input_path,
        special_tokens=special_tokens,
    )

    vocab = init_vocab(
        special_tokens=special_tokens,
    )

    vocab, merges = train_bpe_merge_loop(
        pretoken_counts=pretoken_counts,
        vocab=vocab,
        final_vocab_size=vocab_size,
    )

    return vocab, merges


if __name__ == '__main__': 

    special_tokens = [END_OF_TEXT_TOKEN]
    vocab, merge = train_bpe(
        TINYSTORIES_VALID_PATH, 
        800,
        special_tokens
    )
    
    print(vocab)
    print(merge)