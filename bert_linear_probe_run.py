# bert_linear_probe_run.py
import os, sys, math, argparse
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump

# --- project root resolution so you can run from subfolders ---
_this_file = os.path.abspath(__file__)
_project_root = os.path.dirname(_this_file)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    LABEL_MAP,
    PROMPT_TRAIN_CSV_MINI,
)


def resolve(p):
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(_project_root, p.lstrip("./"))
    return p


PROMPT_TRAIN_CSV = resolve(PROMPT_TRAIN_CSV_MINI)
PROMPT_VAL_CSV = resolve(PROMPT_VAL_CSV_MINI)


parser = argparse.ArgumentParser()
parser.add_argument("--encoder", default="bert-base-uncased", help="encoder model id")
parser.add_argument(
    "--use_subset", action="store_true", help="use subset for quick test"
)
parser.add_argument("--subset_train", type=int, default=2000)
parser.add_argument("--subset_val", type=int, default=2000)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--save_path", default="./bert_outputs/linear_probe.joblib")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| Encoder:", args.encoder)

tokenizer = AutoTokenizer.from_pretrained(args.encoder)
model = AutoModel.from_pretrained(args.encoder).to(DEVICE)
model.eval()


def encode_texts(texts, batch_size=32, max_len=256):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            last = out.last_hidden_state  # [B,L,H]
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (last * mask).sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = (summed / denom).cpu().numpy()
            vecs.append(pooled)
    return np.vstack(vecs)


# load data
print("Loading CSVs...")
train_df = pd.read_csv(PROMPT_TRAIN_CSV)
val_df = pd.read_csv(PROMPT_VAL_CSV)  # full val by default

if args.use_subset:
    train_df = train_df.sample(n=min(args.subset_train, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(args.subset_val, len(val_df)), random_state=42)

print("Train rows:", len(train_df), "Val rows:", len(val_df))

# encode
print("Encoding train...")
X_train = encode_texts(
    train_df["text"].astype(str).tolist(), batch_size=args.batch, max_len=args.max_len
)
y_train = train_df["label"].astype(int).to_numpy()
print("Encoding val...")
X_val = encode_texts(
    val_df["text"].astype(str).tolist(), batch_size=args.batch, max_len=args.max_len
)
y_val = val_df["label"].astype(int).to_numpy()

# scale & fit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Training logistic regression...")
clf = LogisticRegression(
    max_iter=2000, multi_class="multinomial", solver="lbfgs", class_weight="balanced"
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("\n=== Linear-probe Classification Report ===")
print(
    classification_report(
        y_val,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(y_val, y_pred, labels=list(LABEL_MAP.keys())),
)

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
dump({"clf": clf, "scaler": scaler}, args.save_path)
print("Saved classifier+scaler to:", args.save_path)
