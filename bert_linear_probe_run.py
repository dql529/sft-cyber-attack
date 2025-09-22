# bert_linear_probe_run.py —— 无 argparse，训练 linear-probe 并保存 meta
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from nlp_shared import rp, PREP_VERSION, apply_cleaning, load_encoder, encode_texts

# ---- 项目配置 ----
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    PROMPT_VAL_CSV_MEDIUM,
    LABEL_MAP,
)

# 选择 train/val CSV 路径（更稳健的写法）
# 确保 nlp_shared.rp 已导入
# ---- 开关与参数（按需改动）----
USE_TRAIN_MINI = True
USE_TRAIN_MEDIUM = False
USE_VAL_MINI = True
USE_VAL_MEDIUM = False
# ---- end ----

# 防止同时多开
if sum([bool(USE_TRAIN_MINI), bool(USE_TRAIN_MEDIUM)]) > 1:
    raise RuntimeError(
        "Train selection ambiguous: only one of USE_TRAIN_MINI/USE_TRAIN_MEDIUM may be True."
    )
if sum([bool(USE_VAL_MINI), bool(USE_VAL_MEDIUM)]) > 1:
    raise RuntimeError(
        "Val selection ambiguous: only one of USE_VAL_MINI/USE_VAL_MEDIUM may be True."
    )

# train csv selection
if USE_TRAIN_MINI:
    train_csv = rp(PROMPT_TRAIN_CSV_MINI)
elif USE_TRAIN_MEDIUM:
    train_csv = rp(PROMPT_TRAIN_CSV_MEDIUM)
else:
    train_csv = rp(PROMPT_TRAIN_CSV)

# val csv selection
if USE_VAL_MINI:
    val_csv = rp(PROMPT_VAL_CSV_MINI)
elif USE_VAL_MEDIUM:
    val_csv = rp(PROMPT_VAL_CSV_MEDIUM)
else:
    val_csv = rp(PROMPT_VAL_CSV)

# 日志打印
print(f"[CONFIG] train_csv = {train_csv}")
print(f"[CONFIG] val_csv   = {val_csv}")

print(f"[CFG] ENCODER={ENCODER}  MAX_LEN={MAX_LEN}  CLEANING={CLEANING_MODE_TRAIN}")

# 读数据
tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)

# 文本清洗（训练/验证必须一致）
tr_texts = apply_cleaning(tr["text"].astype(str).tolist(), CLEANING_MODE_TRAIN)
va_texts = apply_cleaning(va["text"].astype(str).tolist(), CLEANING_MODE_TRAIN)
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()

# 编码
tok, mdl, device = load_encoder(ENCODER)
print(f"[ENC] device={device}")
print("[ENC] Encoding train ...")
X_tr = encode_texts(tr_texts, tok, mdl, device, max_len=MAX_LEN, batch_size=BATCH)
print("[ENC] Encoding val ...")
X_va = encode_texts(va_texts, tok, mdl, device, max_len=MAX_LEN, batch_size=BATCH)

# 训练
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_va_n = scaler.transform(X_va)

clf = LogisticRegression(
    max_iter=2000, multi_class="multinomial", solver="lbfgs", class_weight="balanced"
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

# 保存 + meta
meta = {
    "encoder": ENCODER,
    "max_len": MAX_LEN,
    "pooling": "mean",
    "cleaning_mode": CLEANING_MODE_TRAIN,
    "prep_version": PREP_VERSION,
    "script": "bert_linear_probe_run.py",
}
dump({"clf": clf, "scaler": scaler, "meta": meta}, rp(SAVE_PATH))
print("[SAVE] model+scaler+meta ->", rp(SAVE_PATH))
print("[META]", meta)
