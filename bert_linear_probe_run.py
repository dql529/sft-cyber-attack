# bert_linear_probe_run.py  —— 无 argparse 版
import os, sys, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# ---------- 基本配置 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    LABEL_MAP,
)


def rp(p):
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


# 开关 & 参数
USE_TRAIN_MINI = True
USE_VAL_MINI = True
ENCODER = "bert-base-uncased"
BATCH = 16
MAX_LEN = 256
SAVE_PATH = "./bert_outputs/linear_probe.joblib"
# -----------------------------

train_csv = rp(PROMPT_TRAIN_CSV_MINI if USE_TRAIN_MINI else PROMPT_TRAIN_CSV)
val_csv = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
os.makedirs(rp("./bert_outputs"), exist_ok=True)

print(f"[DATA] TRAIN={train_csv}")
print(f"[DATA] VAL  ={val_csv}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CFG] ENCODER={ENCODER}  DEVICE={DEVICE}")

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    t = m.group(1).strip() if m else s
    return " ".join(t.split())


# 读数据 & 清洗
tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)
tr_texts = tr["text"].astype(str).apply(clean_text).tolist()
va_texts = va["text"].astype(str).apply(clean_text).tolist()
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()

# 编码器
tok = AutoTokenizer.from_pretrained(ENCODER)
model = AutoModel.from_pretrained(ENCODER).to(DEVICE)
model.eval()


def encode(texts):
    vecs = []
    for i in range(0, len(texts), BATCH):
        b = texts[i : i + BATCH]
        enc = tok(
            b, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc, return_dict=True).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pooling
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)


print("[RUN] Encoding train ...")
X_tr = encode(tr_texts)
print("[RUN] Encoding val ...")
X_va = encode(va_texts)

# 标准化 + LR
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_va_n = scaler.transform(X_va)

print("[RUN] Fitting LogisticRegression ...")
clf = LogisticRegression(
    max_iter=2000, multi_class="multinomial", solver="lbfgs", class_weight="balanced"
)
clf.fit(X_tr_n, y_tr)

y_pred = clf.predict(X_va_n)
print("\n=== Linear-probe Report ===")
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

dump({"clf": clf, "scaler": scaler}, rp(SAVE_PATH))
print("Saved model to:", rp(SAVE_PATH))
