# bert_linear_probe_run.py —— 统一开关 + 智能缓存 + 权重策略 + 保存 meta
import os, sys
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from nlp_shared import rp, PREP_VERSION, apply_cleaning, load_encoder, encode_texts

# ---------------- 基础配置（来自 config.py） ----------------
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    PROMPT_VAL_CSV_MEDIUM,
    LABEL_MAP,
)

# ==================== 顶部开关（你只改这里） ====================
DATA_TIER = "mini"  # 只改这一处： "mini" | "medium" | "full"
ENCODER = "bert-base-uncased"
MAX_LEN = 256
BATCH = 16
CLEANING_MODE_TRAIN = "task_only"  # "raw_prompt" | "task_only"

# 分类器策略（可按需改）：
WEIGHT_MODE = "none"  # "none" | "balanced" | "pow0.5" | "pow0.7"
MULTI_CLASS = "multinomial"  # "multinomial" | "ovr"
SOLVER = "lbfgs"  # "lbfgs"(multi) | "saga"(multi/ovr) | "liblinear"(ovr)
MAX_ITER = 2000

# 保存位置（留 None 自动带参命名；若你想固定路径可写字符串）
SAVE_DIR = "./bert_outputs"
SAVE_PATH = None  # None => 自动命名 linear_probe__{tag}.joblib
USE_EMBED_CACHE = True
# ===========================================================


# -------------- 路径选择（按 DATA_TIER 自动选） --------------
def pick_csvs(tier: str):
    tier = str(tier).lower()
    if tier == "mini":
        return rp(PROMPT_TRAIN_CSV_MINI), rp(PROMPT_VAL_CSV_MINI), "mini"
    elif tier == "medium":
        return rp(PROMPT_TRAIN_CSV_MEDIUM), rp(PROMPT_VAL_CSV_MEDIUM), "medium"
    elif tier == "full":
        return rp(PROMPT_TRAIN_CSV), rp(PROMPT_VAL_CSV), "full"
    else:
        raise ValueError(f"Unknown DATA_TIER: {tier}")


train_csv, val_csv, tier_tag = pick_csvs(DATA_TIER)


# -------------- 命名工具：缓存/模型带参区分 --------------
def sanitize(s: str) -> str:
    return (
        str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "")
    )


enc_tag = sanitize(ENCODER)
clean_tag = "raw" if CLEANING_MODE_TRAIN == "raw_prompt" else "task"
train_bn = os.path.splitext(os.path.basename(train_csv))[0]
val_bn = os.path.splitext(os.path.basename(val_csv))[0]
# 作为 key：档位 + encoder + 长度 + 清洗 + 具体文件名（包含 seed 时自动区分）
KEY = f"{tier_tag}__{enc_tag}__L{MAX_LEN}__{clean_tag}__{train_bn}__{val_bn}"

os.makedirs(rp(SAVE_DIR), exist_ok=True)

# 缓存文件名（不会再互相覆盖）
CACHE_X_TRAIN = rp(f"{SAVE_DIR}/X_train__{KEY}.npy")
CACHE_X_VAL = rp(f"{SAVE_DIR}/X_val__{KEY}.npy")

# 模型保存名（自动命名）
if SAVE_PATH is None:
    SAVE_PATH = rp(f"{SAVE_DIR}/linear_probe__{KEY}.joblib")
else:
    SAVE_PATH = rp(SAVE_PATH)

# ------------------------ 日志 ------------------------
print(f"[CONFIG] DATA_TIER={DATA_TIER}  train_csv={train_csv}")
print(f"[CONFIG] val_csv  ={val_csv}")
print(
    f"[CFG] ENCODER={ENCODER}  MAX_LEN={MAX_LEN}  BATCH={BATCH}  CLEANING={CLEANING_MODE_TRAIN}"
)
print(
    f"[CFG] WEIGHT_MODE={WEIGHT_MODE}  MULTI_CLASS={MULTI_CLASS}  SOLVER={SOLVER}  MAX_ITER={MAX_ITER}"
)
print(f"[PATH] SAVE={SAVE_PATH}")
print(f"[CACHE] X_train={CACHE_X_TRAIN}")
print(f"[CACHE] X_val  ={CACHE_X_VAL}")

# ------------------------ 读数据 ------------------------
print("[I/O] Loading CSVs ...")
tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)
print(f"[I/O] train rows: {len(tr)}  val rows: {len(va)}")

# ------------------------ 清洗 ------------------------
print("[PREP] Applying cleaning ...")
tr_texts = apply_cleaning(tr["text"].astype(str).tolist(), CLEANING_MODE_TRAIN)
va_texts = apply_cleaning(va["text"].astype(str).tolist(), CLEANING_MODE_TRAIN)
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()

# ------------------------ 编码（带缓存校验） ------------------------
X_tr, X_va = None, None
if USE_EMBED_CACHE and os.path.exists(CACHE_X_TRAIN) and os.path.exists(CACHE_X_VAL):
    try:
        print("[CACHE] Loading cached embeddings ...")
        tmp_tr = np.load(CACHE_X_TRAIN)
        tmp_va = np.load(CACHE_X_VAL)
        if tmp_tr.shape[0] == len(tr) and tmp_va.shape[0] == len(va):
            X_tr, X_va = tmp_tr, tmp_va
            print(f"[CACHE] Loaded X_tr.shape={X_tr.shape}  X_va.shape={X_va.shape}")
        else:
            print(
                f"[CACHE] Shape mismatch: cache {tmp_tr.shape[0]}/{tmp_va.shape[0]} vs csv {len(tr)}/{len(va)}; will re-encode."
            )
            X_tr = X_va = None
    except Exception as e:
        print("[CACHE] Failed to load cache; will re-encode. Error:", e)
        X_tr = X_va = None

if X_tr is None or X_va is None:
    print("[ENC] Loading encoder & model ...")
    tok, mdl, device = load_encoder(ENCODER)
    print(f"[ENC] device={device}")
    print("[ENC] Encoding train ...")
    X_tr = encode_texts(tr_texts, tok, mdl, device, max_len=MAX_LEN, batch_size=BATCH)
    print("[ENC] Encoding val ...")
    X_va = encode_texts(va_texts, tok, mdl, device, max_len=MAX_LEN, batch_size=BATCH)
    if USE_EMBED_CACHE:
        try:
            print("[CACHE] Saving embeddings ...")
            np.save(CACHE_X_TRAIN, X_tr)
            np.save(CACHE_X_VAL, X_va)
            print("[CACHE] Saved.")
        except Exception as e:
            print("[CACHE] Failed to save cache:", e)

# ------------------------ 训练/评估 ------------------------
print("[TRAIN] Scaling + training classifier ...")
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_va_n = scaler.transform(X_va)


def make_class_weight(y, mode: str):
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
    raise ValueError(f"Unknown WEIGHT_MODE: {mode}")


class_weight = make_class_weight(y_tr, WEIGHT_MODE)

clf = LogisticRegression(
    max_iter=MAX_ITER, multi_class=MULTI_CLASS, solver=SOLVER, class_weight=class_weight
)
print("[FIT] Fitting LogisticRegression ...")
clf.fit(X_tr_n, y_tr)

y_pred = clf.predict(X_va_n)
print("\n=== Linear-probe Report (train->val) ===")
print(
    classification_report(
        y_va,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n", confusion_matrix(y_va, y_pred, labels=list(LABEL_MAP.keys()))
)

# ------------------------ 保存 + meta ------------------------
meta = {
    "encoder": ENCODER,
    "max_len": MAX_LEN,
    "pooling": "mean",
    "cleaning_mode": CLEANING_MODE_TRAIN,
    "prep_version": PREP_VERSION,
    "script": "bert_linear_probe_run.py",
    "data_tier": DATA_TIER,
    "train_csv": train_csv,
    "val_csv": val_csv,
    "weight_mode": WEIGHT_MODE,
    "multi_class": MULTI_CLASS,
    "solver": SOLVER,
    "key": KEY,
}
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
dump({"clf": clf, "scaler": scaler, "meta": meta}, SAVE_PATH)
print("[SAVE] model+scaler+meta ->", SAVE_PATH)
print("[META]", meta)
