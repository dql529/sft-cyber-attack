# grid_search_linear_probe.py —— 通用 grid search (mini/medium/full)
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

from nlp_shared import rp, PREP_VERSION, apply_cleaning, load_encoder, encode_texts
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_VAL_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV_MEDIUM,
    LABEL_MAP,
)

# ====== 顶部开关 ======
# DATA_TIER = "mini"  # "mini" | "medium" | "full"
# ENCODER_LIST = ["bert-base-uncased"]
# MAX_LEN_LIST = [256]  # 可加 192
# CLEAN_LIST = ["task_only", "raw_prompt"]
# POOL_LIST = ["mean"]  # 可加 "cls"

# WEIGHT_LIST = ["none", "pow0.7", "balanced"]
# MC_SOLVER_LIST = [("ovr", "liblinear"), ("multinomial", "lbfgs")]
# =====================
DATA_TIER = "mini"
ENCODER_LIST = ["bert-base-uncased"]
MAX_LEN_LIST = [256]
CLEAN_LIST = ["task_only"]  # 去掉 raw_prompt
POOL_LIST = ["mean"]
WEIGHT_LIST = ["none", "pow0.7"]  # 足够
MC_SOLVER_LIST = [("multinomial", "lbfgs")]  # 避免 liblinear 弃用告警
# =====================
MAX_ITER = 2000
BATCH = 16
SAVE_DIR = "./bert_outputs"
EXP_DIR = "./experiments"

# 更好读的输出：重写结果文件 + 额外两个摘要 CSV
OVERWRITE_RESULTS = True
RESULTS_CSV = rp(f"{EXP_DIR}/results_linear_probe_{DATA_TIER}.csv")
TOPK_CSV = rp(f"{EXP_DIR}/topk_linear_probe_{DATA_TIER}.csv")
GROUPBEST_CSV = rp(f"{EXP_DIR}/groupbest_linear_probe_{DATA_TIER}.csv")
TOPK = 3
# =====================

os.makedirs(rp(SAVE_DIR), exist_ok=True)
os.makedirs(rp(EXP_DIR), exist_ok=True)
if OVERWRITE_RESULTS and os.path.exists(RESULTS_CSV):
    os.remove(RESULTS_CSV)


# ---- CSV 路径选择 ----
def choose_paths(tier):
    t = str(tier).lower()
    if t == "mini":
        return rp(PROMPT_TRAIN_CSV_MINI), rp(PROMPT_VAL_CSV_MINI)
    if t == "medium":
        return rp(PROMPT_TRAIN_CSV_MEDIUM), rp(PROMPT_VAL_CSV_MEDIUM)
    if t == "full":
        return rp(PROMPT_TRAIN_CSV), rp(PROMPT_VAL_CSV)
    raise ValueError(f"bad DATA_TIER={tier}")


# ---- 命名工具 ----
def sanitize(s):
    return (
        str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "")
    )


def key_for(enc, max_len, clean, pool, ts, train_csv, val_csv):
    enc_tag = sanitize(enc)
    clean_tag = "raw" if clean == "raw_prompt" else "task"
    pool_tag = pool
    ts_tag = "L" if ts == "left" else "R"
    bn_tr = os.path.splitext(os.path.basename(train_csv))[0]
    bn_va = os.path.splitext(os.path.basename(val_csv))[0]
    return f"{DATA_TIER}__{enc_tag}__L{max_len}__{clean_tag}__{pool_tag}__ts{ts_tag}__prep{PREP_VERSION}__{bn_tr}__{bn_va}"


# ---- 类权重策略 ----
def class_weight_from(y, mode):
    mode = str(mode).lower()
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode.startswith("pow"):
        p = float(mode.replace("pow", ""))
        cnt = Counter(y)
        n = len(y)
        k = len(cnt)
        return {c: (n / (k * cnt[c])) ** p for c in cnt}
    raise ValueError(f"bad weight mode: {mode}")


# ---- 训练 + 评估 ----
def train_eval_lr(X_tr, y_tr, X_va, y_va, weight_mode, mc, solver):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_va)
    cw = class_weight_from(y_tr, weight_mode)
    clf = LogisticRegression(
        max_iter=MAX_ITER, multi_class=mc, solver=solver, class_weight=cw
    )
    t0 = time.time()
    clf.fit(Xtr, y_tr)
    fit_s = time.time() - t0
    y_pred = clf.predict(Xva)
    acc = accuracy_score(y_va, y_pred)
    macro = f1_score(
        y_va, y_pred, labels=list(LABEL_MAP.keys()), average="macro", zero_division=0
    )
    wavg = f1_score(
        y_va, y_pred, labels=list(LABEL_MAP.keys()), average="weighted", zero_division=0
    )
    return clf, scaler, acc, macro, wavg, fit_s


# ---- 主程序 ----
def main():
    train_csv, val_csv = choose_paths(DATA_TIER)
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    y_tr = df_tr["label"].astype(int).to_numpy()
    y_va = df_va["label"].astype(int).to_numpy()

    all_rows = []
    for enc in ENCODER_LIST:
        tok = mdl = device = None
        for max_len in MAX_LEN_LIST:
            for clean in CLEAN_LIST:
                # 清洗文本
                tr_texts = apply_cleaning(df_tr["text"].astype(str).tolist(), clean)
                va_texts = apply_cleaning(df_va["text"].astype(str).tolist(), clean)

                # 在 tr_texts / va_texts 生成之后加
                def describe(name, arr):
                    lens = [len(x) for x in arr]
                    short_ratio = sum(l < 30 for l in lens) / len(lens)
                    print(
                        f"[CHECK] {name}: min={min(lens)}, avg={sum(lens)/len(lens):.1f}, max={max(lens)}, short(<30)={short_ratio:.2%}"
                    )

                describe("task_only.train", tr_texts)
                describe("task_only.val", va_texts)
                print("[SAMPLE CLEAN] ", va_texts[0][:200])

                # raw_prompt 保尾部；task_only 保开头
                ts = "left" if clean == "raw_prompt" else "right"

                for pool in POOL_LIST:
                    KEY = key_for(enc, max_len, clean, pool, ts, train_csv, val_csv)
                    cache_tr = rp(f"{SAVE_DIR}/X_train__{KEY}.npy")
                    cache_va = rp(f"{SAVE_DIR}/X_val__{KEY}.npy")

                    # 缓存
                    X_tr = X_va = None
                    if os.path.exists(cache_tr) and os.path.exists(cache_va):
                        try:
                            X_tr = np.load(cache_tr)
                            X_va = np.load(cache_va)
                            if X_tr.shape[0] != len(df_tr) or X_va.shape[0] != len(
                                df_va
                            ):
                                print(f"[CACHE] mismatch {KEY}, re-encode.")
                                X_tr = X_va = None
                            else:
                                print(f"[CACHE] hit {KEY}")
                        except Exception:
                            X_tr = X_va = None

                    # 编码
                    if X_tr is None or X_va is None:
                        if tok is None:
                            tok, mdl, device = load_encoder(enc)
                        print(
                            f"[ENC] {enc}, L={max_len}, clean={clean}, pool={pool}, ts={ts}, device={device}"
                        )
                        X_tr = encode_texts(
                            tr_texts,
                            tok,
                            mdl,
                            device,
                            max_len=max_len,
                            batch_size=BATCH,
                            pooling=pool,
                            truncation_side=ts,
                        )
                        X_va = encode_texts(
                            va_texts,
                            tok,
                            mdl,
                            device,
                            max_len=max_len,
                            batch_size=BATCH,
                            pooling=pool,
                            truncation_side=ts,
                        )
                        np.save(cache_tr, X_tr)
                        np.save(cache_va, X_va)
                        print(f"[CACHE] saved {cache_tr}, {cache_va}")

                    # 扫线性头
                    for weight_mode in WEIGHT_LIST:
                        for mc, solver in MC_SOLVER_LIST:
                            print(f"[FIT] {KEY} | weight={weight_mode} | {mc}/{solver}")
                            clf, scaler, acc, macro, wavg, fit_s = train_eval_lr(
                                X_tr, y_tr, X_va, y_va, weight_mode, mc, solver
                            )
                            rec = {
                                "data_tier": DATA_TIER,
                                "train_csv": train_csv,
                                "val_csv": val_csv,
                                "encoder": enc,
                                "max_len": max_len,
                                "cleaning": clean,
                                "pooling": pool,
                                "ts": ts,
                                "weight_mode": weight_mode,
                                "multi_class": mc,
                                "solver": solver,
                                "acc": round(acc, 4),
                                "macro_f1": round(macro, 4),
                                "weighted_f1": round(wavg, 4),
                                "fit_sec": round(fit_s, 2),
                                "n_train": len(df_tr),
                                "n_val": len(df_va),
                                "key": KEY,
                            }
                            all_rows.append(rec)

    # 写主结果（重写，按宏F1/Acc 降序）
    res = pd.DataFrame(all_rows)
    order = [
        "data_tier",
        "encoder",
        "max_len",
        "cleaning",
        "pooling",
        "ts",
        "weight_mode",
        "multi_class",
        "solver",
        "acc",
        "macro_f1",
        "weighted_f1",
        "fit_sec",
        "n_train",
        "n_val",
        "train_csv",
        "val_csv",
        "key",
    ]
    res = res[order].sort_values(by=["macro_f1", "acc"], ascending=[False, False])
    res.to_csv(RESULTS_CSV, index=False, float_format="%.4f")
    print(f"\n[RESULTS] saved -> {RESULTS_CSV}")

    # TopK
    top = res.head(TOPK).copy()
    top.to_csv(TOPK_CSV, index=False, float_format="%.4f")
    print(f"[TOPK] saved -> {TOPK_CSV}")
    print("\n=== TOP by Macro‑F1 ===")
    print(
        top[
            [
                "encoder",
                "max_len",
                "cleaning",
                "pooling",
                "ts",
                "weight_mode",
                "multi_class",
                "solver",
                "acc",
                "macro_f1",
                "weighted_f1",
            ]
        ]
    )

    # 每个 (encoder,max_len,cleaning,pooling,ts) 组内选最优头
    grp_cols = ["encoder", "max_len", "cleaning", "pooling", "ts"]
    gb = (
        res.sort_values(by=["macro_f1", "acc"], ascending=[False, False])
        .groupby(grp_cols, as_index=False)
        .head(1)
    )
    gb[
        grp_cols
        + [
            "weight_mode",
            "multi_class",
            "solver",
            "acc",
            "macro_f1",
            "weighted_f1",
            "fit_sec",
            "key",
        ]
    ].to_csv(GROUPBEST_CSV, index=False, float_format="%.4f")
    print(f"[GROUPBEST] saved -> {GROUPBEST_CSV}")


if __name__ == "__main__":
    main()
