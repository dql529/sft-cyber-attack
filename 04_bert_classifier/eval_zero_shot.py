# eval_zero_shot_batch.py  （替换你原脚本）
import os, sys, math, pandas as pd, torch
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# project-relative import (assumes config.py 在上级目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import PROMPT_VAL_CSV_MINI, PROMPT_VAL_CSV, LABEL_MAP

# choose validation file (mini by default)
VAL_CSV = PROMPT_VAL_CSV_MINI  # change to PROMPT_VAL_CSV to run full set
print("Using validation file:", VAL_CSV)

df = pd.read_csv(VAL_CSV)
texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int).tolist()

# device id for pipeline: int (0 means cuda:0), -1 means CPU
device_id = 0 if torch.cuda.is_available() else -1
print("torch.cuda.is_available():", torch.cuda.is_available(), "device_id:", device_id)

# create zero-shot pipeline once
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device_id
)

candidate_labels = [v for k, v in LABEL_MAP.items()]


# batch inference helper
def batched_zero_shot(inputs, labels, batch_size=16):
    """
    inputs: list of strings
    labels: candidate labels
    yields: predicted label strings in same order
    """
    preds = []
    n = len(inputs)
    for i in tqdm(range(0, n, batch_size), desc="Batched zero-shot"):
        batch = inputs[i : i + batch_size]
        # pipeline supports list input and batch_size param
        res = classifier(
            batch, candidate_labels=labels, multi_label=False, batch_size=len(batch)
        )
        # when input is list, res is a list of dicts
        for r in res:
            preds.append(r["labels"][0])
    return preds


# pick batch size — tune depending on GPU memory
BATCH_SIZE = 32
pred_labels = batched_zero_shot(texts, candidate_labels, batch_size=BATCH_SIZE)

label2id = {v: k for k, v in LABEL_MAP.items()}
y_pred = [label2id.get(p, -1) for p in pred_labels]

print("\n=== Zero-shot classification report ===")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys())))

# save detailed outputs for debugging
out_df = pd.DataFrame(
    {
        "text": texts,
        "gold_id": y_true,
        "gold_name": [LABEL_MAP[i] for i in y_true],
        "pred_name": pred_labels,
        "pred_id": y_pred,
    }
)
out_csv = os.path.join(PROJECT_ROOT, "eval_reports_zero_shot.csv")
out_df.to_csv(out_csv, index=False, quoting=1)
print("Saved detailed predictions to:", out_csv)
