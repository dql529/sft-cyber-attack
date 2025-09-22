# nlp_shared.py
import os, re, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 版本号（用于将来升级兼容判断）
PREP_VERSION = "1.0.0"

# ---------- 路径工具 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def rp(p: str) -> str:
    """Resolve relative project path to absolute."""
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(ROOT, p.lstrip("./"))
    return p


# ---------- 文本清洗（更鲁棒） ----------
# 兼容大小写、CRLF；允许以 'Answer:'、'---' 或文件结尾为终止
_TASK_RE = re.compile(
    r"\[task\].*?input:\s*(.*?)(?:\r?\n\s*answer:|\r?\n\s*---|\Z)", re.S | re.I
)


def extract_task_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _TASK_RE.search(text)
    if m:
        return " ".join(m.group(1).strip().split())
    # 没匹配上就返回压缩后的原文（裁 512）
    return " ".join(text.split())[:512]


def apply_cleaning(texts, cleaning_mode: str, min_len: int = 30):
    """
    cleaning_mode in {"task_only", "raw_prompt"}
    - task_only: 仅保留 [Task] Input 的流量描述；若文本过短(<min_len)回退 raw
    - raw_prompt: 原文，压缩多余空白
    """
    if cleaning_mode not in {"task_only", "raw_prompt"}:
        raise ValueError(f"Unknown cleaning_mode: {cleaning_mode}")
    cleaned = []
    for t in texts:
        if cleaning_mode == "task_only":
            c = extract_task_description(t)
            if len(c) < min_len:
                c = " ".join((t if isinstance(t, str) else "").split())[:512]
        else:
            c = " ".join((t if isinstance(t, str) else "").split())[:512]
        cleaned.append(c)
    return cleaned


# ---------- BERT 编码（支持 mean/cls + 截断侧） ----------
def load_encoder(encoder_name: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(encoder_name)
    mdl = AutoModel.from_pretrained(encoder_name).to(device)
    mdl.eval()
    return tok, mdl, device


@torch.no_grad()
def encode_texts(
    texts,
    tokenizer,
    model,
    device,
    max_len: int = 256,
    batch_size: int = 16,
    pooling: str = "mean",  # "mean" | "cls"
    truncation_side: str = "right",  # "right"=保开头 | "left"=保尾部
):
    """
    Encode texts into embeddings.
    - pooling: "mean" (默认) 或 "cls"
    - truncation_side: "right"（默认，保开头）; "left"（保尾部；raw_prompt 场景常用）
    """
    tokenizer.truncation_side = truncation_side
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
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True)
        last = out.last_hidden_state  # [B, L, H]

        if pooling == "cls":
            pooled = last[:, 0, :]  # [CLS]
        else:
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)
