from utils import _read_jsonl
    
# calculate the max len of the original dataset
Path = "./data/processed/train.jsonl"
max_len_src = 0
max_len_tgt = 0
samples = _read_jsonl(Path)
for sample in samples:
    src_len = len(sample["src_ids"])
    tgt_len = len(sample["tgt_ids"])
    if src_len > max_len_src:
        max_len_src = src_len
    if tgt_len > max_len_tgt:
        max_len_tgt = tgt_len
print(f"Max length of source sentences: {max_len_src}")
print(f"Max length of target sentences: {max_len_tgt}")