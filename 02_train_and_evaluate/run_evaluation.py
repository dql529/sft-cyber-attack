# -*- coding: utf-8 -*-
import os, sys, re, pandas as pd, torch

# 关键修复：在导入torch之前，通过环境变量彻底禁用Inductor和Dynamo
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM

# from peft import PeftModel   # 不需要加载 peft，如果你不打算测试 LoRA，请注释或删除这行
from sklearn.metrics import classification_report
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import *


def resolve_path(p):
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(project_root, p)
    return p


BASE_MODEL_PATH = resolve_path(BASE_MODEL_PATH)
OUTPUT_DIR = resolve_path(OUTPUT_DIR)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# load base model (原始模型)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(
    device
)

# 不加载 LoRA adapter，直接用 base 作为 model
model = base
model.eval()

label2id = {v: k for k, v in LABEL_MAP.items()}
labels_list = list(LABEL_MAP.values())
labels_set = {lab.lower(): lab for lab in labels_list}  # 规范化对照

_strip_punct = r""" \t\r\n,.;:!?'\"()[]{}<>"""


def extract(pred_text: str) -> str:
    # 只取 Answer: 之后的第一段
    seg = pred_text.split("Answer:", 1)[-1].strip()
    if not seg:
        return ""
    first_line = seg.splitlines()[0].strip()
    tok = first_line.split(" ")[0].strip(_strip_punct)
    key = tok.lower()
    return labels_set.get(key, "")


df = pd.read_csv(PROMPT_VAL_CSV, nrows=200)
y_true, y_pred = [], []

# 打印前5个样本，便于人工核验
PRINT_N = 5
shown = 0

with torch.inference_mode():
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        prompt = row["text"]  # 已包含 "Answer:" 前缀
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=3,  # 足够输出一个类名
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # ★ 关键：用 token 级切片拿到生成尾部，避免字符串长度不一致的偏移
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        gen_tail = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_label = extract(gen_tail)
        y_pred.append(label2id.get(pred_label, -1))
        y_true.append(int(row["label"]))

        if shown < PRINT_N:
            print("\n--- Sample", i, "---")
            print("[PROMPT][-120:]:", prompt[-120:])
            print("[GEN_TAIL]:", gen_tail)
            print("[PRED]:", pred_label, "| [GOLD]:", LABEL_MAP[int(row["label"])])
            shown += 1

report = classification_report(
    y_true,
    y_pred,
    labels=list(LABEL_MAP.keys()),
    target_names=list(LABEL_MAP.values()),
    zero_division=0,
)
print("\n=== Classification Report ===")
print(report)
