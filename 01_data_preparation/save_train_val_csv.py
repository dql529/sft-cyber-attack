# save_train_val_csv_fixed.py
import pandas as pd
import os
import sys
import csv
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 路径配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import FILTERED_DATA_CSV, PROMPT_TRAIN_CSV, PROMPT_VAL_CSV, LABEL_MAP


def resolve_path(path_str):
    if isinstance(path_str, str) and (
        path_str.startswith("./") or path_str.startswith("../")
    ):
        return os.path.join(project_root, path_str)
    return path_str


FILTERED_DATA_CSV = resolve_path(FILTERED_DATA_CSV)
PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)
# --- 路径配置结束 ---

df = pd.read_csv(FILTERED_DATA_CSV)
print(f"Loaded {len(df)} rows from {FILTERED_DATA_CSV}")

# --- 必要列检查，避免 KeyError ---
required_cols = {
    "proto",
    "state",
    "service",
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "sload",
    "dload",
    "tcprtt",
    "ct_srv_src",
    "attack_cat",
}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns in input CSV: {missing}")

print("🔧 Engineering new semantic features from numerical data...")

# ====================================================================
# 分箱要处理的列（和你原来一致）
cols_to_categorize = [
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "dur",
    "sload",
    "dload",
    "tcprtt",
]
# 可选的标签阶层（最多）
base_labels = ["low", "medium", "high", "very_high"]
# ====================================================================

for col in cols_to_categorize:
    # 确保 numeric
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    non_zero_data = df[df[col] > 0][col]
    if not non_zero_data.empty:
        try:
            # 先用 qcut 获取 bins（可能会因为重复值而减少箱数）
            _, bins = pd.qcut(
                non_zero_data, q=len(base_labels), retbins=True, duplicates="drop"
            )
            n_bins = len(bins) - 1
            real_labels = base_labels[:n_bins]  # 若 qcut 把箱数降了，则 labels 截断匹配
            # 应用分箱到整列
            df[f"{col}_cat"] = pd.cut(
                df[col], bins=bins, labels=real_labels, include_lowest=True
            )
            # 把缺失（0 或 NaN）设为 'zero' 并把列设成 categorical，保证 dtype 一致
            cat_dtype = pd.CategoricalDtype(
                categories=real_labels + ["zero"], ordered=True
            )
            df[f"{col}_cat"] = (
                df[f"{col}_cat"]
                .cat.add_categories("zero")
                .fillna("zero")
                .astype(cat_dtype)
            )
            print(f"  - Categorized '{col}' into {n_bins} bins + zero.")
        except Exception as e:
            # 兜底：如果 qcut/pd.cut 都失败，返回原始数值的字符串化类别（小概率）
            print(f"  - Could not categorize '{col}': {e}. Fallback to coarse bins.")
            # 简单分为 zero / positive 两类
            cat_dtype = pd.CategoricalDtype(
                categories=["zero", "positive"], ordered=False
            )
            df[f"{col}_cat"] = pd.Categorical(
                ["positive" if v > 0 else "zero" for v in df[col]], dtype=cat_dtype
            )
    else:
        # 全为 0 的列，直接统一为 categorical 'zero'
        cat_dtype = pd.CategoricalDtype(categories=base_labels + ["zero"], ordered=True)
        df[f"{col}_cat"] = pd.Categorical(["zero"] * len(df), dtype=cat_dtype)
        print(f"  - Column '{col}' contains only zeros; set '{col}_cat' = 'zero'.")

# LABEL 映射（假设 config.LABEL_MAP 是 {int: str}）
label2id = {v: k for k, v in LABEL_MAP.items()}

# attack_cat 处理：去两端空白并统一大小写（若有 NaN 保持）
df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
df["label"] = df["attack_cat"].map(label2id)

num_null_labels = df["label"].isnull().sum()
if num_null_labels > 0:
    print(f"  - 警告：丢弃 {num_null_labels} 行无法映射攻击类别的数据。")
    df = df.dropna(subset=["label"]).copy()

df["label"] = df["label"].astype(int)


# ========== 构造 prompt 的函数（更鲁棒） ==========
def safe_get(row, key, default="-"):
    v = row.get(key, default)
    if pd.isna(v):
        return default
    return v


def build_prompt(row):
    class_labels = "[Backdoor, DoS, Exploits, Normal, Reconnaissance, Shellcode, Worms]"

    proto = safe_get(row, "proto", "unknown")
    state = safe_get(row, "state", "unknown")
    dur_cat = safe_get(row, "dur_cat", "zero")

    description_parts = [
        f"A {proto} connection with {state} state had a {dur_cat} duration."
    ]

    service = safe_get(row, "service", "-")
    if service != "-" and service != "unknown":
        description_parts.append(f"The connection service was identified as {service}.")
    else:
        description_parts.append("The connection service was not specified.")

    # packet and bytes categories (使用 *_cat 列)
    s_pkts = safe_get(row, "spkts_cat", "zero")
    s_bytes = safe_get(row, "sbytes_cat", "zero")
    d_pkts = safe_get(row, "dpkts_cat", "zero")
    d_bytes = safe_get(row, "dbytes_cat", "zero")
    description_parts.append(
        f"It involved {s_pkts} source packets (total size: {s_bytes}) and {d_pkts} destination packets (total size: {d_bytes})."
    )

    sload = safe_get(row, "sload_cat", "zero")
    dload = safe_get(row, "dload_cat", "zero")
    description_parts.append(
        f"The source load was {sload} and the destination load was {dload}."
    )

    # 如果原始 tcprtt 数值大于0，添加 RTT 描述（注意使用原始数值判断）
    try:
        tcprtt_val = float(row.get("tcprtt", 0) or 0)
    except Exception:
        tcprtt_val = 0
    if proto == "tcp" and tcprtt_val > 0:
        tcprtt_cat = safe_get(row, "tcprtt_cat", "zero")
        description_parts.append(f"The TCP round-trip time was {tcprtt_cat}.")

    # 访问历史次数（避免非数值）
    ct_srv_src = int(row.get("ct_srv_src", 0) or 0)
    description_parts.append(
        f"The source has connected to this same service {ct_srv_src} times before."
    )

    traffic_description = " ".join(description_parts)

    # 使用三引号构建 prompt（注意保留结构，但不要在 text 内部加入多余的引号）
    prompt = f"""You are an expert cybersecurity analyst. Your task is to classify network 
      traffic into ONE of the following categories: {class_labels}.

   Follow these rules strictly:
   1. Analyze the traffic description.
   2. Your response MUST be ONLY the category name.
   3. Do not add any explanation, punctuation, or other text.

   ---
   [Example]
   Input: A tcp connection with FIN state had a normal duration. It involved low source packets
      (total size: low) and zero destination packets (total size: zero). The source previously accessed
      the service 1 times.
   Answer: Normal
   ---
   [Task]
   Input: {traffic_description}
   Answer:"""

    return prompt


# ----------------- 安全版：生成 prompt 并记录错误 -----------------
from tqdm import tqdm
import traceback
import time

# 先确保 ct_srv_src 是整数（避免 int("nan") 报错等）
if "ct_srv_src" in df.columns:
    df["ct_srv_src"] = (
        pd.to_numeric(df["ct_srv_src"], errors="coerce").fillna(0).astype(int)
    )
else:
    df["ct_srv_src"] = 0

# 也统一 service 字段为字符串，防止 None/float 导致拼接异常
if "service" in df.columns:
    df["service"] = df["service"].astype(str).fillna("-")
else:
    df["service"] = "-"

# 准备容器与错误记录
texts = []
errors = []  # list of tuples (index, error_str, tb_short)
start_time = time.time()

# 用迭代并捕获异常的方式逐行生成（比 apply 更易调试）
for i, (idx, row) in enumerate(
    tqdm(df.iterrows(), total=len(df), desc="Building prompts")
):
    try:
        txt = build_prompt(row)
        # 防守：保证是字符串
        if not isinstance(txt, str):
            txt = str(txt)
        texts.append(txt)
    except Exception as e:
        # 捕获完整 trace，记录到 errors 列表并继续
        tb = traceback.format_exc(limit=5)
        errors.append((idx, str(e), tb))
        texts.append("")  # 用空字符串占位，后面可筛掉或检查
        # 现场打印前几条错误以便快速定位
        if len(errors) <= 10:
            print(f"\n--- build_prompt ERROR at row idx={idx} (sample) ---")
            print("Error:", e)
            print(tb)
            print(
                "Row sample keys:",
                {
                    k: row.get(k)
                    for k in [
                        "proto",
                        "state",
                        "service",
                        "dur",
                        "ct_srv_src",
                        "attack_cat",
                    ]
                },
            )
        # 可选：如果错误很频繁并且你想中断以立刻修复，取消下面注释
        # if len(errors) > 200: break

elapsed = time.time() - start_time
print(
    f"\nPrompt generation finished in {elapsed:.1f}s. Total rows: {len(df)}, errors: {len(errors)}"
)

# 将生成文本赋回 DataFrame
df["text"] = texts

# 如果有错误，写一个错误日志文件（包含行索引与 traceback）
if errors:
    os.makedirs(os.path.dirname(PROMPT_TRAIN_CSV), exist_ok=True)
    err_log_path = os.path.join(
        os.path.dirname(PROMPT_TRAIN_CSV), "prompt_generation_errors.log"
    )
    with open(err_log_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt generation errors: {len(errors)}\n\n")
        for idx, errstr, tb in errors:
            f.write(f"ROW_INDEX: {idx}\nERROR: {errstr}\nTRACEBACK:\n{tb}\n{'-'*60}\n")
    print(f"Errors logged to: {err_log_path}")
    # 打印前 5 个有问题行的索引，便于你直接定位数据源
    print("Sample error rows indices:", [e[0] for e in errors[:10]])

# 简单检查：有多少是空 prompt
num_empty = df["text"].astype(str).str.strip().eq("").sum()
print(f"Empty/failed prompts: {num_empty} / {len(df)}")

# 如果大量空，可以把出问题的行导出供你直接查看
if num_empty > 0:
    bad_idx = df[df["text"].astype(str).str.strip() == ""].index
    sample_bad = df.loc[
        bad_idx[:50], ["proto", "state", "service", "dur", "ct_srv_src", "attack_cat"]
    ]
    sample_bad_path = os.path.join(
        os.path.dirname(PROMPT_TRAIN_CSV), "prompt_failed_sample.csv"
    )
    sample_bad.to_csv(sample_bad_path, index=True, quoting=csv.QUOTE_ALL)
    print(f"Saved sample of failed rows to: {sample_bad_path}")

# ----------------- 之后继续原有切分与保存流程 -----------------
# ========== 划分训练/验证集（检查 stratify 条件） ==========
label_counts = Counter(df["label"].tolist())
print("Label distribution (after filtering):", dict(label_counts))

min_count = min(label_counts.values()) if label_counts else 0
if min_count < 2:
    print(
        "  - 警告：某些类别样本太少，无法做 stratified split. 将使用无 stratify 的随机切分。"
    )
    stratify_arg = None
else:
    stratify_arg = df["label"]

train_df, val_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    random_state=42,
    stratify=stratify_arg,
)

# ========== 保存 CSV（强制引用所有字段，避免 prompt 内部的逗号/换行破坏 CSV 结构） ==========
os.makedirs(os.path.dirname(PROMPT_TRAIN_CSV), exist_ok=True)
train_df.to_csv(PROMPT_TRAIN_CSV, index=False, quoting=csv.QUOTE_ALL)
val_df.to_csv(PROMPT_VAL_CSV, index=False, quoting=csv.QUOTE_ALL)

print("保存新的、信息更丰富的语义化Prompt CSV文件：")
print(f"- {PROMPT_TRAIN_CSV}  (rows: {len(train_df)})")
print(f"- {PROMPT_VAL_CSV}    (rows: {len(val_df)})")
