# svm_on_bert_embeddings.py —— 无 argparse 版
import os, sys, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
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


USE_TRAIN_MINI = True
USE_VAL_MINI = True
ENCODER = "bert-base-uncased"
BATCH = 16
MAX_LEN = 256
KERNEL = "linear"  # "linear" or "rbf"
C_VAL = 1.0
SAVE_PATH = "./svm_outputs/svm_bert.joblib"
# -----------------------------

train_csv = rp(PROMPT_TRAIN_CSV_MINI if USE_TRAIN_MINI else PROMPT_TRAIN_CSV)
val_csv = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
os.makedirs(rp("./svm_outputs"), exist_ok=True)
print(f"[DATA] TRAIN={train_csv}  VAL={val_csv}  KERNEL={KERNEL} C={C_VAL}")

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    t = m.group(1).strip() if m else s
    return " ".join(t.split())


tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)
tr_texts = tr["text"].astype(str).apply(clean_text).tolist()
va_texts = va["text"].astype(str).apply(clean_text).tolist()
y_tr = tr["label"].astype(int).to_numpy()
y_va = va["label"].astype(int).to_numpy()

tok = AutoTokenizer.from_pretrained(ENCODER)
model = AutoModel.from_pretrained(ENCODER).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()
DEVICE = next(model.parameters()).device


def encode(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        b = texts[i : i + BATCH]
        enc = tok(
            b, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc, return_dict=True).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


print("[RUN] Encoding train ...")
X_tr = encode(tr_texts)
print("[RUN] Encoding val ...")
X_va = encode(va_texts)

sc = StandardScaler()
X_tr_n = sc.fit_transform(X_tr)
X_va_n = sc.transform(X_va)

if KERNEL == "linear":
    clf = LinearSVC(C=C_VAL, class_weight="balanced", max_iter=10000, dual=False)
else:
    clf = SVC(kernel="rbf", C=C_VAL, class_weight="balanced")

print("[RUN] Fitting SVM ...")
clf.fit(X_tr_n, y_tr)
y_pred = clf.predict(X_va_n)

print("\n=== SVM Report ===")
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

dump({"clf": clf, "scaler": sc}, rp(SAVE_PATH))
print("Saved model to:", rp(SAVE_PATH))
