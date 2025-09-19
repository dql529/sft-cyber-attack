# svm_on_bert_embeddings.py
import os, sys, argparse, re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

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
        if p.startswith("./") or p.startswith("../")
        else p
    )


ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="bert-base-uncased")
ap.add_argument("--use_train_mini", action="store_true")
ap.add_argument("--use_val_mini", action="store_true")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
ap.add_argument("--C", type=float, default=1.0)
args = ap.parse_args()

train_csv = rp(PROMPT_TRAIN_CSV_MINI if args.use_train_mini else PROMPT_TRAIN_CSV)
val_csv = rp(PROMPT_VAL_CSV_MINI if args.use_val_mini else PROMPT_VAL_CSV)
print("Train:", train_csv, " Val:", val_csv)

tok = AutoTokenizer.from_pretrained(args.encoder)
model = AutoModel.from_pretrained(args.encoder).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()
DEVICE = next(model.parameters()).device

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s):
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    return " ".join((m.group(1).strip() if m else s).split())


def encode(texts):
    outs = []
    for i in range(0, len(texts), args.batch):
        b = texts[i : i + args.batch]
        enc = tok(
            b,
            truncation=True,
            padding=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            outs.append(pooled.cpu().numpy())
    return np.vstack(outs)


tr = pd.read_csv(train_csv)
va = pd.read_csv(val_csv)
Xtr = encode(tr["text"].astype(str).apply(clean_text).tolist())
ytr = tr["label"].astype(int).to_numpy()
Xva = encode(va["text"].astype(str).apply(clean_text).tolist())
yva = va["label"].astype(int).to_numpy()

sc = StandardScaler()
Xtrn = sc.fit_transform(Xtr)
Xvan = sc.transform(Xva)

if args.kernel == "linear":
    clf = LinearSVC(C=args.C, class_weight="balanced", max_iter=10000, dual=False)
else:
    clf = SVC(kernel="rbf", C=args.C, class_weight="balanced")

clf.fit(Xtrn, ytr)
yp = clf.predict(Xvan)

print(
    classification_report(
        yva,
        yp,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(yva, yp, labels=list(LABEL_MAP.keys())))
os.makedirs(rp("./svm_outputs"), exist_ok=True)
dump({"clf": clf, "scaler": sc}, rp("./svm_outputs/svm_bert.joblib"))
