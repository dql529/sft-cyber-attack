# hybrid_agent.py —— 无 argparse，自动按 meta 对齐（训练/推理一致）
import os, sys, numpy as np, pandas as pd, torch
from joblib import load
from transformers import pipeline

from nlp_shared import rp, PREP_VERSION, apply_cleaning, load_encoder, encode_texts
from config import PROMPT_VAL_CSV, PROMPT_VAL_CSV_MINI, LABEL_MAP

# ---- 开关与参数 ----
USE_VAL_MINI = True
BERT_JOBLIB = "./bert_outputs/linear_probe.joblib"

# “auto” = 使用 joblib.meta；也可手写为具体值，但建议 auto，避免不一致
ENCODER = "auto"  # or "bert-base-uncased"
MAX_LEN = "auto"  # or 256
CLEANING_MODE = "auto"  # or "task_only" / "raw_prompt"

# 兜底阈值/LLM
TAU = 0.85
LLM_MODEL = "facebook/bart-large-mnli"
LLM_BATCH = 16
SAVE_FALLBACK_CSV = "./bert_outputs/fallback_samples.csv"
# --------------------

VAL_CSV = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
print(f"[CFG] VAL={VAL_CSV}")
print(f"[CFG] joblib={BERT_JOBLIB}  τ={TAU}  LLM={LLM_MODEL}")

# 加载 joblib + meta
bundle = load(rp(BERT_JOBLIB))
clf = bundle["clf"]
scaler = bundle["scaler"]
meta = bundle.get("meta", {})

print("[META from joblib]", meta)
# 解析/对齐 meta
enc_name = meta.get("encoder") if ENCODER == "auto" else ENCODER
max_len = meta.get("max_len") if MAX_LEN == "auto" else MAX_LEN
clean_mode = meta.get("cleaning_mode") if CLEANING_MODE == "auto" else CLEANING_MODE

# 严格校验（防错用）
if meta:
    if enc_name != meta.get("encoder"):
        raise RuntimeError(
            f"Encoder mismatch: joblib={meta.get('encoder')} vs run={enc_name}"
        )
    if max_len != meta.get("max_len"):
        raise RuntimeError(
            f"max_len mismatch: joblib={meta.get('max_len')} vs run={max_len}"
        )
    if clean_mode != meta.get("cleaning_mode"):
        raise RuntimeError(
            f"cleaning_mode mismatch: joblib={meta.get('cleaning_mode')} vs run={clean_mode}"
        )
else:
    print(
        "[WARN] joblib has no meta; proceed but ensure settings match training manually."
    )

print(f"[ALIGN] encoder={enc_name}  max_len={max_len}  cleaning={clean_mode}")

# 读取数据并做与训练一致的清洗
df = pd.read_csv(VAL_CSV)
texts_raw = df["text"].astype(str).tolist()
labels_true = df["label"].astype(int).to_numpy()
texts = apply_cleaning(texts_raw, clean_mode)

# BERT 编码（按 meta 对齐）
tok, mdl, device = load_encoder(enc_name)
print(f"[ENC] device={device}")
X = encode_texts(texts, tok, mdl, device, max_len=max_len, batch_size=16)
Xn = scaler.transform(X)

# 先用 BERT+LR 预测
if hasattr(clf, "predict_proba"):
    proba = clf.predict_proba(Xn)
    y_hat = proba.argmax(1)
    conf = proba.max(1)
else:
    scores = clf.decision_function(Xn)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    y_hat = scores.argmax(1)
    conf = scores.max(1)
    print("[WARN] classifier has no predict_proba; conf is uncalibrated.")

# 找低置信样本
need = np.where(conf < TAU)[0].tolist()
print(f"[RUN] low-confidence (<{TAU}): {len(need)} / {len(texts)}")

# LLM 兜底（批量）
pred_final = y_hat.copy()
if need:
    cand_labels = [v for _, v in LABEL_MAP.items()]
    zshot = pipeline(
        "zero-shot-classification",
        model=LLM_MODEL,
        device=(0 if torch.cuda.is_available() else -1),
    )
    # 分批调用，避免“sequential”警告
    records = []
    for i in range(0, len(need), LLM_BATCH):
        idxs = need[i : i + LLM_BATCH]
        batch_inputs = [texts[j] for j in idxs]
        results = zshot(batch_inputs, cand_labels, multi_label=False)
        # HF pipeline 对单个/多个输入返回格式不同，统一成 list
        if isinstance(results, dict):
            results = [results]
        for k, j in enumerate(idxs):
            pred_name = results[k]["labels"][0]
            score = float(results[k]["scores"][0])
            pred_id = {v: k for k, v in LABEL_MAP.items()}.get(pred_name, pred_final[j])
            records.append(
                {
                    "index": j,
                    "raw_text": texts_raw[j],
                    "clean_text": texts[j],
                    "bert_conf": float(conf[j]),
                    "bert_pred": int(y_hat[j]),
                    "llm_pred_name": pred_name,
                    "llm_pred_score": score,
                    "final_pred": int(pred_id),
                }
            )
            pred_final[j] = pred_id
    # 保存兜底细节，便于分析
    os.makedirs(rp("./bert_outputs"), exist_ok=True)
    import pandas as pd

    pd.DataFrame(records).to_csv(rp(SAVE_FALLBACK_CSV), index=False)
    print("[SAVE] fallback samples ->", rp(SAVE_FALLBACK_CSV))

# 指标
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Hybrid Agent (BERT primary + LLM fallback) ===")
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
