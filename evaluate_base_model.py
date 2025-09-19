# evaluate_base_model_small.py
import os, torch, pandas as pd, numpy as np, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from config import BASE_MODEL_PATH, PROMPT_VAL_CSV, LABEL_MAP

# =========== Config ===========
MAX_SAMPLES = 200  # <-- 先跑前 N 条，快速验证
MAX_NEW_TOKENS = 6  # 生成 token 数（调小减少续写模板噪声）
VERBOSE_PRINT_N = 5  # 打印前几个样例
SAVE_DIR = "./eval_reports_small"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# =================================

# mappings
label2id = {v: k for k, v in LABEL_MAP.items()}
id2label = {k: v for k, v in LABEL_MAP.items()}
label_names = [lab for lab in LABEL_MAP.values()]
label_names_lc = [l.lower() for l in label_names]
label_re = re.compile(
    r"\b(" + "|".join(re.escape(l) for l in label_names_lc) + r")\b",
    flags=re.IGNORECASE,
)

# load tokenizer + model
print("Loading tokenizer and base model from:", BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# If OOM: consider torch_dtype=torch.float16 in the from_pretrained call
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model.to(device)
model.eval()

_strip_punct = r""" \t\r\n,.;:!?'\"()[]{}<>"""


def extract(pred_text: str) -> str:
    """Robust extract: strict Answer: first token -> fallback keyword regex."""
    if not isinstance(pred_text, str):
        pred_text = str(pred_text)
    # Strict: check after "Answer:"
    seg = pred_text.split("Answer:", 1)
    if len(seg) > 1:
        tail = seg[1].strip()
        if tail:
            first_line = tail.splitlines()[0].strip()
            tok = first_line.split(" ")[0].strip(_strip_punct)
            key = tok.lower()
            for lab in label_names_lc:
                if key == lab:
                    return lab.title()
    # Fallback: regex find any label word
    m = label_re.search(pred_text.lower())
    if m:
        return m.group(1).title()
    return ""  # unknown


# Load validation csv
print("Loading val CSV:", PROMPT_VAL_CSV)
df = pd.read_csv(PROMPT_VAL_CSV)
if "text" not in df.columns or "label" not in df.columns:
    raise RuntimeError("验证 CSV 必须包含 'text' 和 'label' 两列。")

n_total = len(df)
n = min(MAX_SAMPLES, n_total)
print(f"Using first {n} samples out of {n_total} for quick eval.")

rows_out = []
y_true = []
y_pred = []

with torch.inference_mode():
    for i, (idx, row) in enumerate(
        tqdm(df.iloc[:n].iterrows(), total=n, desc="Evaluating small set")
    ):
        prompt = str(row["text"])
        # run generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # keep generated tail only
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        gen_tail = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # optionally truncate generation before template markers to reduce noise
        for stop_tok in ["---", "[", "Task", "Example"]:
            pos = gen_tail.find(stop_tok)
            if pos != -1:
                gen_tail = gen_tail[:pos]
                break

        pred_name = extract(gen_tail)
        pred_id = label2id.get(pred_name, -1)
        gold_id = int(row["label"])
        gold_name = id2label.get(gold_id, str(gold_id))

        y_true.append(gold_id)
        y_pred.append(pred_id)

        rows_out.append(
            {
                "index": idx,
                "prompt_snippet": prompt[-300:],
                "gen_tail": gen_tail,
                "pred_name": pred_name,
                "pred_id": pred_id,
                "gold_name": gold_name,
                "gold_id": gold_id,
            }
        )

        if i < VERBOSE_PRINT_N:
            print("\n--- Sample", i, f"(index {idx}) ---")
            print("[PROMPT][-300:]", prompt[-300:])
            print("[GEN_TAIL]:", repr(gen_tail))
            print("[PRED]:", pred_name, "| [GOLD]:", gold_name)

# metrics & save
labels_sorted = sorted(list(LABEL_MAP.keys()))
target_names = [LABEL_MAP[i] for i in labels_sorted]

report = classification_report(
    y_true, y_pred, labels=labels_sorted, target_names=target_names, zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

print("\n=== Classification Report (small sample) ===")
print(report)
print("Confusion matrix:\n", cm)

# save full CSV of predictions for inspection
pred_df = pd.DataFrame(rows_out)
pred_csv = os.path.join(SAVE_DIR, "predictions_small.csv")
pred_df.to_csv(pred_csv, index=False, quoting=1)  # csv.QUOTE_ALL numeric 1
print("Saved detailed predictions to:", pred_csv)

# save textual report
report_txt = os.path.join(SAVE_DIR, "classification_report_small.txt")
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("Classification Report (small sample)\n\n")
    f.write(report)
    f.write("\n\nConfusion matrix:\n")
    np.savetxt(f, cm, fmt="%d")
print("Saved report to:", report_txt)
