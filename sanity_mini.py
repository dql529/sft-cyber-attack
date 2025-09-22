# sanity_mini.py —— 单条最佳组合，完全不走缓存
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from nlp_shared import rp, apply_cleaning, load_encoder, encode_texts
from config import PROMPT_TRAIN_CSV_MINI, PROMPT_VAL_CSV_MINI, LABEL_MAP

ENCODER = "bert-base-uncased"
MAX_LEN = 256
BATCH = 16
CLEAN = "task_only"
TS = "right"  # 保开头
POOL = "mean"

train_csv = rp(PROMPT_TRAIN_CSV_MINI)
val_csv = rp(PROMPT_VAL_CSV_MINI)

print("[I/O]", train_csv, val_csv)
tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)

tr_texts = apply_cleaning(tr["text"].astype(str).tolist(), CLEAN)
va_texts = apply_cleaning(va["text"].astype(str).tolist(), CLEAN)
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()


# 简单健康检查
def describe(name, arr):
    lens = [len(x) for x in arr]
    print(
        f"[CHECK] {name}: min={min(lens)}, avg={sum(lens)/len(lens):.1f}, max={max(lens)}"
    )


describe("task_only.train", tr_texts)
describe("task_only.val", va_texts)
print("[SAMPLE]", va_texts[0][:200])

tok, mdl, device = load_encoder(ENCODER)
print("[ENC] device:", device)

# 不使用缓存，直接编码
X_tr = encode_texts(
    tr_texts,
    tok,
    mdl,
    device,
    max_len=MAX_LEN,
    batch_size=BATCH,
    pooling=POOL,
    truncation_side=TS,
)
X_va = encode_texts(
    va_texts,
    tok,
    mdl,
    device,
    max_len=MAX_LEN,
    batch_size=BATCH,
    pooling=POOL,
    truncation_side=TS,
)

scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_va_n = scaler.transform(X_va)

clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
clf.fit(X_tr_n, y_tr)
y_pred = clf.predict(X_va_n)

print("\n=== Sanity (mini, single best combo) ===")
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
