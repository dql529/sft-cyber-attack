# eval_full_val_with_linear_probe.py —— 无 argparse，按 meta 对齐评估
import os, sys, numpy as np, pandas as pd
from joblib import load

from nlp_shared import rp, PREP_VERSION, apply_cleaning, load_encoder, encode_texts
from config import PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, LABEL_MAP

# ---- 开关 ----
USE_VAL_MINI = False
BERT_JOBLIB = "./bert_outputs/linear_probe.joblib"
# -------------

VAL_CSV = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
print(f"[CFG] VAL={VAL_CSV}  joblib={BERT_JOBLIB}")

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
