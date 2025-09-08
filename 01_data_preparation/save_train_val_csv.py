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
# ==============================================================================
# === 修改 1: 在列表中加入 sload, dload, tcprtt ===
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
# ==============================================================================

labels = ["low", "medium", "high", "very_high"]
for col in cols_to_categorize:
    non_zero_data = df[df[col] > 0][col]
    if not non_zero_data.empty:
        try:
            # 使用 qcut 进行分箱
            quantiles, bins = pd.qcut(
                non_zero_data,
                q=len(labels),
                labels=labels,
                retbins=True,
                duplicates="drop",
            )
            # 应用分箱到整个列
            df[f"{col}_cat"] = pd.cut(
                df[col], bins=bins, labels=labels, include_lowest=True
            )
            # 将0值和NaN值处理为 'zero' 类别
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
# 转换label列，并处理无法映射的情况
df["label"] = df["attack_cat"].map(label2id)
num_null_labels = df["label"].isnull().sum()
if num_null_labels > 0:
    print(f"  - 警告：丢弃 {num_null_labels} 行无法映射攻击类别的数据。")
    df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)


# ==============================================================================
# === 修改 2: 升级 build_prompt 函数 ===
def build_prompt(row):
    """
    构建一个结构化、带示例的提示，以强制模型按固定格式输出。
    (新版：包含更多特征和逻辑判断)
    """
    class_labels = "[Backdoor, DoS, Exploits, Normal, Reconnaissance, Shellcode, Worms]"

    # 使用一个列表来动态构建描述，更灵活、更清晰
    description_parts = [
        f"A {row['proto']} connection with {row['state']} state had a {row['dur_cat']} duration."
    ]

    # 智能处理 'service' 字段
    if row['service'] != '-' and pd.notna(row['service']):
        description_parts.append(f"The connection service was identified as {row['service']}.")
    else:
        description_parts.append("The connection service was not specified.")

    # 添加数据包信息
    description_parts.append(
        f"It involved {row['spkts_cat']} source packets (total size: {row['sbytes_cat']}) "
        f"and {row['dpkts_cat']} destination packets (total size: {row['dbytes_cat']})."
    )

    # 添加负载（速率）信息
    description_parts.append(
        f"The source load was {row['sload_cat']} and the destination load was {row['dload_cat']}."
    )

    # 仅当协议是 tcp 且 tcprtt > 0 时，才添加 TCP RTT 信息
    if row['proto'] == 'tcp' and row['tcprtt'] > 0:
        description_parts.append(f"The TCP round-trip time was {row['tcprtt_cat']}.")

    # 添加历史访问信息，并微调措辞
    description_parts.append(f"The source has connected to this same service {row['ct_srv_src']} times before.")

    # 将所有描述部分用空格连接成一个完整的字符串
    traffic_description = " ".join(description_parts)

    # 使用三重引号构建一个清晰、多行的 f-string 模板
    prompt = f'''You are an expert cybersecurity analyst. Your task is to classify network 
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
   Answer:'''

    return prompt
# ==============================================================================


df["text"] = df.apply(build_prompt, axis=1)

train_df, val_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
)

os.makedirs(os.path.dirname(PROMPT_TRAIN_CSV), exist_ok=True)
train_df.to_csv(PROMPT_TRAIN_CSV, index=False)
val_df.to_csv(PROMPT_VAL_CSV, index=False)

print("保存新的、信息更丰富的语义化Prompt CSV文件：")
print(f"- {PROMPT_TRAIN_CSV}")
print(f"- {PROMPT_VAL_CSV}")

# 打印一个样本以供检查
print("\n--- 生成的Prompt样本 ---")
print(train_df.iloc[0]['text'])
print("--------------------------")