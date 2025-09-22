# hybrid_agent.py  —— 无 argparse 版本（代码内开关）
import os, sys, re
import numpy as np
import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- 基本路径与配置 ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    LABEL_MAP,
)


def rp(p: str) -> str:
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


# ============= 代码内开关（自行修改） =============
# 是否使用 mini 验证集
USE_VAL_MINI = True  # True 用 PROMPT_VAL_CSV_MINI，False 用 PROMPT_VAL_CSV

# 模型与运行参数
ENCODER = "bert-base-uncased"  # BERT 编码器
BERT_JOBLIB = "./bert_outputs/linear_probe.joblib"  # 之前保存好的 LR 分类器与 scaler
BATCH = 16  # 编码 batch size
MAX_LEN = 256  # 编码最大长度
TAU = 0.85  # 置信度阈值：低于此阈值走 LLM 兜底
LLM_MODEL = "facebook/bart-large-mnli"  # zero-shot NLI 模型
LLM_EXPLAIN = False  # 低置信样本是否生成一句“理由”（可选，较慢）
# ===============================================

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_ID = 0 if torch.cuda.is_available() else -1

# 选择验证集
VAL_CSV = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)

# 标签映射
ID2NAME = LABEL_MAP
NAME2ID = {v: k for k, v in LABEL_MAP.items()}

# 打印配置
print(f"[CFG] Device={DEVICE}  Encoder={ENCODER}")
print(f"[CFG] VAL_CSV={VAL_CSV}")
print(f"[CFG] TAU={TAU}  LLM_MODEL={LLM_MODEL}  Explain={LLM_EXPLAIN}")
print(f"[CFG] BERT_JOBLIB={BERT_JOBLIB}")

# ---------------- 加载 BERT 分类器（线性探针） ----------------
bundle = load(rp(BERT_JOBLIB))  # 期望 {"clf": ..., "scaler": ...}
clf = bundle["clf"]
scaler = bundle["scaler"]

# ---------------- 文本清洗（仅保留 [Task] Input 段） ----------------
TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def extract_task_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = TASK_RE.search(text)
    if m:
        return " ".join(m.group(1).strip().split())
    return " ".join(text.split())[:512]


# ---------------- 初始化编码器 ----------------
tok = AutoTokenizer.from_pretrained(ENCODER)
enc_model = AutoModel.from_pretrained(ENCODER).to(DEVICE)
enc_model.eval()


def encode_batch(texts):
    vecs = []
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
            out = enc_model(**enc, return_dict=True)
            last = out.last_hidden_state  # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)


# ---------------- LLM 兜底（zero-shot NLI） ----------------
candidate_labels = [v for _, v in LABEL_MAP.items()]
zshot = pipeline("zero-shot-classification", model=LLM_MODEL, device=DEV_ID)


def llm_fallback(texts):
    """
    对低置信文本调用 NLI，返回 [(pred_name, score, rationale), ...]
    """
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
        if LLM_EXPLAIN:
            # 简单一行解释（注意：使用的是同一模型做生成，成本较高，可按需关闭）
            expl_ids = zshot.model.generate(
                **zshot.tokenizer(
                    f"Explain in one sentence why the label '{pred}' fits this traffic:\n\n{t}\n\nAnswer:",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(zshot.device),
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=zshot.tokenizer.eos_token_id,
            )
            expl = zshot.tokenizer.decode(expl_ids[0], skip_special_tokens=True)
            rationale = expl.split("Answer:")[-1].strip()
        out.append((pred, score, rationale))
    return out


# ---------------- 读取数据 & 推理 ----------------
df = pd.read_csv(VAL_CSV)
texts_raw = df["text"].astype(str).tolist()
labels_true = df["label"].astype(int).tolist()
texts = [extract_task_description(t) for t in texts_raw]

print("[RUN] Encoding with BERT ...")
X = encode_batch(texts)
Xn = scaler.transform(X)

print("[RUN] Predicting with BERT+LR ...")
if hasattr(clf, "predict_proba"):
    proba = clf.predict_proba(Xn)
    y_hat = proba.argmax(1)
    conf = proba.max(1)
else:
    scores = clf.decision_function(Xn)
    if scores.ndim == 1:  # binary fallback
        scores = np.vstack([-scores, scores]).T
    y_hat = scores.argmax(1)
    conf = scores.max(1)  # 未校准
    print("[WARN] classifier 无 predict_proba，使用 decision_function，置信度未校准。")

# 低置信度样本 → LLM 兜底
need_fallback_idx = np.where(conf < TAU)[0].tolist()
print(f"[RUN] Low-confidence count (<{TAU}): {len(need_fallback_idx)}")

pred_final = y_hat.copy()
if need_fallback_idx:
    llm_inputs = [texts[i] for i in need_fallback_idx]
    zres = llm_fallback(llm_inputs)
    for j, (name, score, rationale) in zip(need_fallback_idx, zres):
        pred_final[j] = NAME2ID.get(name, pred_final[j])
        # 如需把 rationale 记录下来，可自行收集

# ---------------- 指标输出 ----------------
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
