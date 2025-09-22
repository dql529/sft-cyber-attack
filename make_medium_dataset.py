# make_medium_dataset.py
import os, sys
import math
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# config: 根据你的项目结构调整或 import config
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def rp(p):
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


# 输入：如果你已经有完整的 prompt CSV（train+val），建议基于它做下采样
# 我这里假设你有一个合并的 full_prompt.csv（若没有，可把 train+val 合并）
FULL_PROMPT_TRAIN = rp(PROMPT_TRAIN_CSV)  # 或者你合并后的文件路径
# 输出 medium 文件
OUT_TRAIN = rp("./data/prompt_csv/unsw15_prompt_train_medium.csv")
OUT_VAL = rp("./data/prompt_csv/unsw15_prompt_val_medium.csv")

# 目标规模（总样本 = target_total）。改这里
TARGET_TOTAL = 50000  # 推荐：50000（按你 GPU 调整）
VAL_RATIO = 0.20  # 验证集占比

# 最少保留每类数量（防止稀有类被下采样掉）
MIN_PER_CLASS = 200

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Loading full prompt train:", FULL_PROMPT_TRAIN)
df = pd.read_csv(FULL_PROMPT_TRAIN)

# 如果你只有 train/val 分开的两个文件，也可以把他们 concat
# df = pd.concat([pd.read_csv(PROMPT_TRAIN_CSV), pd.read_csv(PROMPT_VAL_CSV)], ignore_index=True)

print("Total rows in source:", len(df))
print("Label distribution in source:", df["label"].value_counts().to_dict())

n_classes = df["label"].nunique()
target_per_class = {
    int(c): max(MIN_PER_CLASS, int(TARGET_TOTAL // n_classes)) for c in range(n_classes)
}

# But respect original class proportions: compute proportion and scale
orig_counts = df["label"].value_counts().to_dict()
total_orig = len(df)
proportions = {c: orig_counts.get(c, 0) / total_orig for c in range(n_classes)}

# compute desired counts per class by proportion but not below MIN_PER_CLASS
desired = {}
remaining = TARGET_TOTAL
for c, prop in proportions.items():
    desire = max(MIN_PER_CLASS, int(round(prop * TARGET_TOTAL)))
    desired[c] = desire
    remaining -= desire

# if rounding made us off, distribute leftover by largest classes
if remaining != 0:
    # sort classes by orig_count desc, add/subtract 1 until sum matches
    classes_sorted = sorted(orig_counts.keys(), key=lambda x: -orig_counts[x])
    idx = 0
    step = 1 if remaining > 0 else -1
    remaining_abs = abs(remaining)
    while remaining_abs > 0:
        cls = classes_sorted[idx % len(classes_sorted)]
        desired[cls] += step
        remaining_abs -= 1
        idx += 1

print(
    "Desired per-class sample counts (sum={}):".format(sum(desired.values())), desired
)

# sample per class
sampled_chunks = []
for cls, cnt in desired.items():
    sub = df[df["label"] == cls]
    if len(sub) <= cnt:
        sampled = sub.copy()
        print(f"Class {cls}: available {len(sub)} <= desired {cnt}, keep all.")
    else:
        sampled = sub.sample(n=cnt, random_state=RANDOM_SEED)
        print(f"Class {cls}: sampled {cnt} from {len(sub)}.")
    sampled_chunks.append(sampled)

df_medium = pd.concat(sampled_chunks, ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED
)  # shuffle

# split into train/val stratified
train_df, val_df = train_test_split(
    df_medium,
    test_size=VAL_RATIO,
    stratify=df_medium["label"],
    random_state=RANDOM_SEED,
)

print("Medium dataset sizes -> train:", len(train_df), " val:", len(val_df))
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)
print("Saved medium train ->", OUT_TRAIN)
print("Saved medium val   ->", OUT_VAL)
