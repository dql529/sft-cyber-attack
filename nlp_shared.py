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


# ---------- 文本清洗 ----------
_TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def extract_task_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _TASK_RE.search(text)
    if m:
        return " ".join(m.group(1).strip().split())
    return " ".join(text.split())[:512]


def apply_cleaning(texts, cleaning_mode: str):
    """
    cleaning_mode in {"task_only", "raw_prompt"}
    - task_only: 仅保留 [Task] Input 的流量描述
    - raw_prompt: 不清洗，原样输入
    """
    if cleaning_mode not in {"task_only", "raw_prompt"}:
        raise ValueError(f"Unknown cleaning_mode: {cleaning_mode}")
    if cleaning_mode == "task_only":
        return [extract_task_description(t) for t in texts]
    else:
        return [t if isinstance(t, str) else "" for t in texts]


# ---------- BERT 编码（mean pooling） ----------
def load_encoder(encoder_name: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(encoder_name)
    mdl = AutoModel.from_pretrained(encoder_name).to(device)
    mdl.eval()
    return tok, mdl, device


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, max_len=256, batch_size=16):
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
        last = out.last_hidden_state  # [B,L,H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B,L,1]
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean
        vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)
