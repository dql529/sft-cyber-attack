# hybrid_agent.py
import os, sys, argparse, re, numpy as np, pandas as pd, torch
from joblib import load
from transformers import AutoTokenizer, AutoModel, pipeline

# -------- resolve project root & load config --------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from config import PROMPT_TRAIN_CSV, PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, LABEL_MAP


def resolve(p):
    return (
        os.path.join(SCRIPT_DIR, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


PROMPT_TRAIN_CSV = resolve(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve(PROMPT_VAL_CSV)
PROMPT_VAL_CSV_MINI = resolve(PROMPT_VAL_CSV_MINI)

# -------- args --------
ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="bert-base-uncased")
ap.add_argument("--bert_joblib", default="./bert_outputs/linear_probe.joblib")
ap.add_argument("--use_mini", action="store_true", help="use mini val for quick test")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument(
    "--tau",
    type=float,
    default=0.85,
    help="confidence threshold for BERT; below this use LLM fallback",
)
ap.add_argument("--llm_model", default="facebook/bart-large-mnli")
ap.add_argument(
    "--explain",
    action="store_true",
    help="also return a brief rationale from LLM for low-confidence cases",
)
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_ID = 0 if torch.cuda.is_available() else -1
VAL_CSV = PROMPT_VAL_CSV_MINI if args.use_mini else PROMPT_VAL_CSV
ID2NAME = LABEL_MAP
NAME2ID = {v: k for k, v in LABEL_MAP.items()}

print(f"Device: {DEVICE}, Encoder: {args.encoder}, Val: {VAL_CSV}")
print("Loading BERT classifier:", args.bert_joblib)
bundle = load(resolve(args.bert_joblib))  # expects {"clf": ..., "scaler": ...}
clf = bundle["clf"]
scaler = bundle["scaler"]


# -------- text cleaning (same as before) --------
def extract_task_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(
        r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)$",
        text,
        flags=re.S,
    )
    if m:
        return " ".join(m.group(1).strip().split())
    return " ".join(text.split())[:512]


# -------- encoder init --------
tok = AutoTokenizer.from_pretrained(args.encoder)
enc_model = AutoModel.from_pretrained(args.encoder).to(DEVICE)
enc_model.eval()


def encode_batch(texts):
    vecs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i : i + args.batch]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = enc_model(**enc, return_dict=True)
            last = out.last_hidden_state  # [B,L,H]
            mask = enc["attention_mask"].unsqueeze(-1)  # [B,L,1]
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)


# -------- LLM fallback (zero-shot NLI) --------
candidate_labels = [v for _, v in LABEL_MAP.items()]
zshot = pipeline("zero-shot-classification", model=args.llm_model, device=DEV_ID)


def llm_fallback(texts):
    # returns list of (pred_name, score, rationale_opt)
    out = []
    for t in texts:
        res = zshot(
            t,
            candidate_labels,
            multi_label=False,
            hypothesis_template="This traffic instance is {}.",
        )
        pred = res["labels"][0]
        score = float(res["scores"][0])
        rationale = ""
        if args.explain:
            # brief explanation prompt（可选，成本小）
            expl = zshot.tokenizer.decode(
                zshot.model.generate(
                    **zshot.tokenizer(
                        f"Explain in one sentence why the label '{pred}' fits this traffic:\n\n{t}\n\nAnswer:",
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(zshot.device),
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=zshot.tokenizer.eos_token_id,
                )[0],
                skip_special_tokens=True,
            )
            rationale = expl.split("Answer:")[-1].strip()
        out.append((pred, score, rationale))
    return out


# -------- load data & run --------
df = pd.read_csv(VAL_CSV)
texts_raw = df["text"].astype(str).tolist()
labels_true = df["label"].astype(int).tolist()
texts = [extract_task_description(t) for t in texts_raw]

print("Encoding with BERT...")
X = encode_batch(texts)
Xn = scaler.transform(X)
print("Predicting with logistic regression...")
if hasattr(clf, "predict_proba"):
    proba = clf.predict_proba(Xn)
    y_hat = proba.argmax(1)
    conf = proba.max(1)
else:
    # fallback if using a model without proba (e.g., SVM without calibration)
    scores = clf.decision_function(Xn)
    if scores.ndim == 1:  # binary
        scores = np.vstack([-scores, scores]).T
    y_hat = scores.argmax(1)
    conf = scores.max(1)
    # NOTE: not calibrated

# decide which need fallback
need_fallback_idx = np.where(conf < args.tau)[0].tolist()
print(f"Low-confidence count (<{args.tau}):", len(need_fallback_idx))

pred_final = y_hat.copy()
if need_fallback_idx:
    llm_inputs = [texts[i] for i in need_fallback_idx]
    zres = llm_fallback(llm_inputs)
    for j, (name, score, rationale) in zip(need_fallback_idx, zres):
        pred_final[j] = NAME2ID.get(name, pred_final[j])

# metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Hybrid Agent Report (BERT primary, LLM fallback) ===")
print(
    classification_report(
        labels_true,
        pred_final,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(labels_true, pred_final, labels=list(LABEL_MAP.keys())),
)
