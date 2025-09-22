# eval_full_val_with_linear_probe.py —— 无 argparse，按 meta 对齐评估
import os, sys, numpy as np, pandas as pd
from joblib import load

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
    TRAIN_CSV = rp(PROMPT_TRAIN_CSV_MINI)
elif USE_TRAIN_MEDIUM:
    TRAIN_CSV = rp(PROMPT_TRAIN_CSV_MEDIUM)
else:
    TRAIN_CSV = rp(PROMPT_TRAIN_CSV)

# val csv selection
if USE_VAL_MINI:
    VAL_CSV = rp(PROMPT_VAL_CSV_MINI)
elif USE_VAL_MEDIUM:
    VAL_CSV = rp(PROMPT_VAL_CSV_MEDIUM)
else:
    VAL_CSV = rp(PROMPT_VAL_CSV)

# 日志打印
print(f"[CONFIG] TRAIN_CSV = {TRAIN_CSV}")
print(f"[CONFIG] val_csv   = {VAL_CSV}")
BERT_JOBLIB = "./bert_outputs/linear_probe.joblib"
# -------------

bundle = load(rp(BERT_JOBLIB))
clf, scaler, meta = bundle["clf"], bundle["scaler"], bundle.get("meta", {})

if not meta:
    raise RuntimeError(
        "joblib has no meta; cannot ensure training/inference alignment."
    )

enc_name = meta["encoder"]
max_len = meta["max_len"]
clean_mode = meta["cleaning_mode"]
print(f"[ALIGN] encoder={enc_name}  max_len={max_len}  cleaning={clean_mode}")

# 读数据 & 清洗（与训练一致）
df = pd.read_csv(VAL_CSV)
texts = apply_cleaning(df["text"].astype(str).tolist(), clean_mode)
y_true = df["label"].astype(int).to_numpy()

# 编码
tok, mdl, device = load_encoder(enc_name)
print(f"[ENC] device={device}")
X = encode_texts(texts, tok, mdl, device, max_len=max_len, batch_size=16)
Xn = scaler.transform(X)

# 评估
y_pred = clf.predict(Xn)
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Linear-probe on VAL ===")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys())),
)
