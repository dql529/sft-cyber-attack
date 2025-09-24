# svm_probe_run.py —— 线性SVM基线（BERT向量 + LinearSVC Pipeline）
import os, sys
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from nlp_shared import rp, apply_cleaning, load_encoder, encode_texts, PREP_VERSION
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_VAL_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV_MEDIUM,
    LABEL_MAP,
)

# ====== 顶部设置 ======
DATA_TIER = "mini"  # "mini" | "medium" | "full"
ENCODER = "bert-base-uncased"
MAX_LEN = 256
BATCH = 16
CLEANING = "task_only"
POOLING = "mean"
TS = "right"  # task_only→保开头
SAVE_DIR = "./bert_outputs"
# =====================

os.makedirs(rp(SAVE_DIR), exist_ok=True)


def choose_paths(tier):
    t = str(tier).lower()
    if t == "mini":
        return rp(PROMPT_TRAIN_CSV_MINI), rp(PROMPT_VAL_CSV_MINI)
    if t == "medium":
        return rp(PROMPT_TRAIN_CSV_MEDIUM), rp(PROMPT_VAL_CSV_MEDIUM)
    if t == "full":
        return rp(PROMPT_TRAIN_CSV), rp(PROMPT_VAL_CSV)
    raise ValueError(f"bad DATA_TIER={tier}")


train_csv, val_csv = choose_paths(DATA_TIER)
print("[I/O]", train_csv, val_csv)

tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)
tr_texts = apply_cleaning(tr["text"].astype(str).tolist(), CLEANING)
va_texts = apply_cleaning(va["text"].astype(str).tolist(), CLEANING)
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()

# 编码
tok, mdl, device = load_encoder(ENCODER)
print(f"[ENC] {ENCODER}, device={device}")
X_tr = encode_texts(
    tr_texts,
    tok,
    mdl,
    device,
    max_len=MAX_LEN,
    batch_size=BATCH,
    pooling=POOLING,
    truncation_side=TS,
)
X_va = encode_texts(
    va_texts,
    tok,
    mdl,
    device,
    max_len=MAX_LEN,
    batch_size=BATCH,
    pooling=POOLING,
    truncation_side=TS,
)

# 线性 SVM Pipeline（StandardScaler + LinearSVC）
pipe = make_pipeline(
    StandardScaler(with_mean=True),  # 对线性模型有益
    LinearSVC(C=1.0, loss="squared_hinge", random_state=11),  # 一般与 LR 接近，可对比
)
print("[FIT] LinearSVC ...")
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_va)

print("\n=== SVM (BERT embeddings + LinearSVC) ===")
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

# 保存
key = f"{DATA_TIER}__svm__{ENCODER}__L{MAX_LEN}__{CLEANING}__{POOLING}__ts{('R' if TS=='right' else 'L')}__prep{PREP_VERSION}"
outp = rp(f"{SAVE_DIR}/svm_probe__{key}.joblib")
meta = {
    "encoder": ENCODER,
    "max_len": MAX_LEN,
    "pooling": POOLING,
    "cleaning_mode": CLEANING,
    "ts": TS,
    "prep_version": PREP_VERSION,
    "script": "svm_probe_run.py",
    "data_tier": DATA_TIER,
    "train_csv": train_csv,
    "val_csv": val_csv,
}
dump({"pipe": pipe, "meta": meta}, outp)
print("[SAVE]", outp)
