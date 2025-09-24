# hybrid_agent.py —— 无 argparse，自动加载 Top1 组合并评估（Pipeline兼容）
import os, sys
import numpy as np, pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from nlp_shared import rp, apply_cleaning, load_encoder, encode_texts
from config import LABEL_MAP, PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, PROMPT_VAL_CSV_MEDIUM

# ====== 选择评测集 ======
DATA_TIER = "medium"  # "mini" | "medium" | "full"
BATCH = 16
# =======================

EXP_DIR = rp("./experiments")
SAVE_DIR = rp("./bert_outputs")
topk_csv = rp(f"{EXP_DIR}/topk_linear_probe_{DATA_TIER}.csv")

if not os.path.exists(topk_csv):
    raise FileNotFoundError(
        f"TopK file not found: {topk_csv}. 先运行 grid_search_linear_probe.py。"
    )

top = pd.read_csv(topk_csv)
if len(top) == 0:
    raise RuntimeError(f"No rows in {topk_csv}")

r = top.iloc[0]
joblib_path = rp(
    f"{SAVE_DIR}/linear_probe__{r['key']}__w{r['weight_mode']}__{r['multi_class']}__{r['solver']}.joblib"
)
if not os.path.exists(joblib_path):
    raise FileNotFoundError(f"Joblib not found: {joblib_path}")

print(f"[MODEL] loading: {joblib_path}")
bundle = load(joblib_path)

# 兼容：优先使用 pipe；否则用 scaler+clf 组装成 pipe
if "pipe" in bundle:
    pipe = bundle["pipe"]
    meta = bundle.get("meta", {})
else:
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    pipe = make_pipeline(scaler, clf)
    meta = bundle.get("meta", {})

encoder = meta.get("encoder", "bert-base-uncased")
max_len = int(meta.get("max_len", 256))
pooling = meta.get("pooling", "mean")
cleaning = meta.get("cleaning_mode", "task_only")
ts = meta.get("ts", "right" if cleaning == "task_only" else "left")

# 选择验证集：优先使用 meta 中的 val_csv；否则按 tier
val_csv = rp(
    meta.get(
        "val_csv",
        {
            "mini": PROMPT_VAL_CSV_MINI,
            "medium": PROMPT_VAL_CSV_MEDIUM,
            "full": PROMPT_VAL_CSV,
        }[DATA_TIER],
    )
)

print(
    f"[CFG] encoder={encoder}  max_len={max_len}  pooling={pooling}  cleaning={cleaning}  ts={ts}"
)
print(f"[DATA] val_csv={val_csv}")

df = pd.read_csv(val_csv)
texts_raw = df["text"].astype(str).tolist()
labels_true = df["label"].astype(int).to_numpy()

texts = apply_cleaning(texts_raw, cleaning)
tok, mdl, device = load_encoder(encoder)
print(f"[ENC] device={device}")

X = encode_texts(
    texts,
    tok,
    mdl,
    device,
    max_len=max_len,
    batch_size=BATCH,
    pooling=pooling,
    truncation_side=ts,
)
y_pred = pipe.predict(X)

print("\n=== Hybrid (BERT linear-probe only) ===")
print(
    classification_report(
        labels_true,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(labels_true, y_pred, labels=list(LABEL_MAP.keys())),
)
