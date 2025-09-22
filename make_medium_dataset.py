# make_subset_dataset.py —— 通用子集采样器（mini/medium），多 seed 多策略选优
import os, sys, math, shutil
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

from nlp_shared import rp
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_VAL_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV_MEDIUM,
)

# ======== 只改这里（主要是 TIER 和策略/目标量）========
TIER = "mini"  # "mini" | "medium"

# 针对不同 TIER 的默认目标量与验证比例
TARGETS = {
    "mini": {"TOTAL": 2800, "VAL_RATIO": 0.25},  # -> 2100/700（接近你之前的设置）
    "medium": {"TOTAL": 50000, "VAL_RATIO": 0.20},  # -> 40k/10k
}

# 采样策略（根据 TIER 推荐默认值；也可手动改）
#   "proportional"：按原始分布（整体 acc 往往更高）
#   "balanced"    ：各类均等（macro-F1 往往更高）
#   "flatten"     ：介于两者，按 p^ALPHA 重新归一（ALPHA∈(0,1)，越小越平衡）
STRATEGY = {"mini": "balanced", "medium": "flatten"}[TIER]
ALPHA = 0.6  # 仅对 flatten 有效

FROM_BOTH = True  # True=用 full train + full val 合并后采样（样本池更大）
MIN_PER_CLASS = 50  # 每类至少保留的下限（防极端）
RANDOM_SEEDS = [11, 22, 33, 44, 55]  # 多试几次，择优
# ===============================================

OUT_DIR = rp("./data/prompt_csv")

# 输出的“标准路径”：训练脚本会直接从这里读取
BEST_PATHS = {
    "mini": (rp(PROMPT_TRAIN_CSV_MINI), rp(PROMPT_VAL_CSV_MINI)),
    "medium": (rp(PROMPT_TRAIN_CSV_MEDIUM), rp(PROMPT_VAL_CSV_MEDIUM)),
}


def load_full():
    tr = pd.read_csv(rp(PROMPT_TRAIN_CSV))
    if FROM_BOTH:
        va = pd.read_csv(rp(PROMPT_VAL_CSV))
        full = pd.concat([tr, va], ignore_index=True)
    else:
        full = tr
    return full


def desired_counts(full_labels, n_total, strategy="proportional", alpha=0.5):
    cnt = Counter(full_labels)
    classes = sorted(cnt.keys())
    probs = np.array([cnt[c] for c in classes], dtype=float)
    probs = probs / probs.sum()

    if strategy == "balanced":
        q = np.ones_like(probs) / len(probs)
    elif strategy == "proportional":
        q = probs
    elif strategy == "flatten":
        q = np.power(probs, alpha)
        q = q / q.sum()
    else:
        raise ValueError("Unknown strategy")

    des = {c: max(MIN_PER_CLASS, int(round(qi * n_total))) for c, qi in zip(classes, q)}
    # 调整总量到 n_total
    diff = n_total - sum(des.values())
    order = np.argsort(-probs)  # 按原分布从大到小分配剩余/回收
    i = 0
    step = 1 if diff > 0 else -1
    left = abs(diff)
    while left > 0:
        c = classes[order[i % len(classes)]]
        des[c] = max(MIN_PER_CLASS, des[c] + step)
        left -= 1
        i += 1
    return des, classes, probs


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sample_once(full_df, desired_map, seed):
    parts = []
    for c, need in desired_map.items():
        sub = full_df[full_df["label"] == c]
        if len(sub) <= need:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=need, random_state=int(seed)))
    df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def save_best_copy(best_train, best_val, tier):
    dst_tr, dst_va = BEST_PATHS[tier]
    os.makedirs(os.path.dirname(dst_tr), exist_ok=True)
    shutil.copyfile(best_train, dst_tr)
    shutil.copyfile(best_val, dst_va)
    print(f"[COPY] -> {dst_tr}")
    print(f"[COPY] -> {dst_va}")


def main():
    tier = TIER
    n_total = TARGETS[tier]["TOTAL"]
    val_ratio = TARGETS[tier]["VAL_RATIO"]

    full = load_full()
    assert {"text", "label"}.issubset(full.columns), "CSV must have columns: text,label"
    print(
        f"[LOAD] full rows={len(full)}  (FROM_BOTH={FROM_BOTH})  TIER={tier}  TARGET_TOTAL={n_total}  VAL_RATIO={val_ratio}"
    )
    print(f"[STRATEGY] {STRATEGY} (ALPHA={ALPHA if STRATEGY=='flatten' else '-'})")

    des_map, classes, orig_prob = desired_counts(
        full["label"].tolist(), n_total, STRATEGY, ALPHA
    )
    print("[PLAN] desired per class:", des_map)

    target_prob = np.array([des_map[c] for c in classes], dtype=float)
    target_prob /= target_prob.sum()

    os.makedirs(OUT_DIR, exist_ok=True)
    candidates = []
    for s in RANDOM_SEEDS:
        df_sub = sample_once(full, des_map, s)
        tr_df, va_df = train_test_split(
            df_sub, test_size=val_ratio, random_state=int(s), stratify=df_sub["label"]
        )
        # 评分：采样总体分布 vs 目标分布
        cnt = Counter(df_sub["label"].tolist())
        sample_prob = np.array([cnt[c] / len(df_sub) for c in classes])
        jsd = js_divergence(sample_prob, target_prob)

        subtag = f"{tier}_s{s}"
        train_out = os.path.join(OUT_DIR, f"unsw15_prompt_train_{subtag}.csv")
        val_out = os.path.join(OUT_DIR, f"unsw15_prompt_val_{subtag}.csv")
        tr_df.to_csv(train_out, index=False)
        va_df.to_csv(val_out, index=False)
        print(f"[SAVED] seed={s} train={len(tr_df)} val={len(va_df)} JSdiv={jsd:.6f}")
        candidates.append((jsd, s, train_out, val_out))

    # 选最优
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    print(f"[BEST] seed={best[1]}  JSdiv={best[0]:.6f}")
    save_best_copy(best[2], best[3], tier)


if __name__ == "__main__":
    main()
