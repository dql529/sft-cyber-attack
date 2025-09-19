# eval_full_val_with_linear_probe.py
import os, sys, re, numpy as np, pandas as pd, torch
from joblib import load
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, LABEL_MAP


def rp(p):
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if p.startswith("./") or p.startswith("../")
        else p
    )


# --- config ---
ENCODER = "bert-base-uncased"
BATCH = 16
MAX_LEN = 256
USE_MINI = False  # <<< 这里开关 mini/全量
VAL_CSV = rp(PROMPT_VAL_CSV_MINI if USE_MINI else PROMPT_VAL_CSV)
JOBLIB_PATH = rp("./bert_outputs/linear_probe.joblib")

print("Val CSV:", VAL_CSV)
print("Load classifier:", JOBLIB_PATH)
bundle = load(JOBLIB_PATH)
clf, scaler = bundle["clf"], bundle["scaler"]

tok = AutoTokenizer.from_pretrained(ENCODER)
model = AutoModel.from_pretrained(ENCODER).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()
DEVICE = next(model.parameters()).device
print("Device:", DEVICE)

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s):
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    t = m.group(1).strip() if m else s
    return " ".join(t.split())


df = pd.read_csv(VAL_CSV)
texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
y_true = df["label"].astype(int).to_numpy()


def encode(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


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

# 导出错分
mis = pd.DataFrame({"text": df["text"], "gold": y_true, "pred": y_pred})
mis = mis[mis["gold"] != mis["pred"]]
mis.to_csv(rp("./bert_outputs/misclassified_val.csv"), index=False)
print("Saved misclassified to ./bert_outputs/misclassified_val.csv")
