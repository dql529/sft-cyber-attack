# eval_full_val_with_linear_probe.py —— 无 argparse 版
import os, sys, re
import numpy as np
import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 基本配置 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, LABEL_MAP


def rp(p):
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


USE_VAL_MINI = False
ENCODER = "bert-base-uncased"
BATCH = 16
MAX_LEN = 256
JOBLIB_PATH = "./bert_outputs/linear_probe.joblib"
# -----------------------------

val_csv = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
bundle = load(rp(JOBLIB_PATH))
clf, scaler = bundle["clf"], bundle["scaler"]

print(f"[DATA] VAL={val_csv}")
print(
    f"[CFG] ENCODER={ENCODER}  DEVICE={'cuda' if torch.cuda.is_available() else 'cpu'}"
)

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    t = m.group(1).strip() if m else s
    return " ".join(t.split())


df = pd.read_csv(val_csv)
texts = df["text"].astype(str).apply(clean_text).tolist()
y_true = df["label"].astype(int).to_numpy()

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


print("[RUN] Encoding val ...")
X = encode(texts)
Xn = scaler.transform(X)
y_pred = clf.predict(Xn)

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

os.makedirs(rp("./bert_outputs"), exist_ok=True)
mis = pd.DataFrame({"text": df["text"], "gold": y_true, "pred": y_pred})
mis = mis[mis["gold"] != mis["pred"]]
mis.to_csv(rp("./bert_outputs/misclassified_val.csv"), index=False)
print("Saved misclassified to ./bert_outputs/misclassified_val.csv")
