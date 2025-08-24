import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# --- 路径配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import FILTERED_DATA_CSV, PROMPT_TRAIN_CSV, PROMPT_VAL_CSV, LABEL_MAP


def resolve_path(path_str):
    if isinstance(path_str, str) and path_str.startswith("./"):
        return os.path.join(project_root, path_str)
    return path_str


FILTERED_DATA_CSV = resolve_path(FILTERED_DATA_CSV)
PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)
# --- 路径配置结束 ---

df = pd.read_csv(FILTERED_DATA_CSV)

print("🔧 Engineering new semantic features from numerical data...")
cols_to_categorize = ["sbytes", "dbytes", "spkts", "dpkts", "dur"]
labels = ["low", "medium", "high", "very_high"]
for col in cols_to_categorize:
    non_zero_data = df[df[col] > 0][col]
    if not non_zero_data.empty:
        try:
            quantiles, bins = pd.qcut(
                non_zero_data,
                q=len(labels),
                labels=labels,
                retbins=True,
                duplicates="drop",
            )
            df[f"{col}_cat"] = pd.cut(
                df[col], bins=bins, labels=labels, include_lowest=True
            )
            df[f"{col}_cat"] = (
                df[f"{col}_cat"].cat.add_categories("zero").fillna("zero")
            )
            print(f"  - Categorized '{col}' into semantic bins.")
        except Exception as e:
            print(
                f"  - Could not categorize '{col}': {e}. Using raw values as fallback."
            )
            df[f"{col}_cat"] = df[col]
    else:
        df[f"{col}_cat"] = "zero"
        print(f"  - Column '{col}' contains only zeros.")

label2id = {v: k for k, v in LABEL_MAP.items()}
df["attack_cat"] = df["attack_cat"].str.strip()
df["label"] = df["attack_cat"].map(label2id)
num_null_labels = df["label"].isnull().sum()
if num_null_labels > 0:
    print(f"  - 警告：丢弃 {num_null_labels} 行无法映射攻击类别的数据。")
    df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)


def build_prompt(row):
    return (
        f"A {row['proto']} connection with {row['state']} state had a {row['dur_cat']} duration. "
        f"It involved {row['spkts_cat']} source packets (total size: {row['sbytes_cat']}) "
        f"and {row['dpkts_cat']} destination packets (total size: {row['dbytes_cat']}). "
        f"The source previously accessed the service {row['ct_srv_src']} times. "
        f"\n\nYou are a classifier. Choose exactly one class from: "
        f"[Backdoor, DoS, Exploits, Normal, Reconnaissance, Shellcode, Worms]. "
        f"Respond with the class name only."
        f"\nQuestion: What attack category is this traffic?\nAnswer:"
    )


df["text"] = df.apply(build_prompt, axis=1)

train_df, val_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
)

os.makedirs(os.path.dirname(PROMPT_TRAIN_CSV), exist_ok=True)
train_df.to_csv(PROMPT_TRAIN_CSV, index=False)
val_df.to_csv(PROMPT_VAL_CSV, index=False)

print("✅ 已保存新的、语义化的Prompt CSV文件：")
print(f"- {PROMPT_TRAIN_CSV}")
print(f"- {PROMPT_VAL_CSV}")
