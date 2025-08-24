import pandas as pd
import os
import sys

# --- 路径配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import RAW_DATA_CSV, FILTERED_DATA_CSV

def resolve_path(path_str):
    if isinstance(path_str, str) and path_str.startswith('./'):
        return os.path.join(project_root, path_str)
    return path_str

RAW_DATA_CSV = resolve_path(RAW_DATA_CSV)
FILTERED_DATA_CSV = resolve_path(FILTERED_DATA_CSV)
# --- 路径配置结束 ---

df = pd.read_csv(RAW_DATA_CSV)

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

df.fillna(0, inplace=True)

selected_cols = [
    "dur", "proto", "state", "service", "spkts", "dpkts", "sbytes", "dbytes",
    "sload", "dload", "tcprtt", "synack", "ackdat", "ct_srv_src",
    "ct_state_ttl", "attack_cat",
]
existing_cols = [col for col in selected_cols if col in df.columns]
df = df[existing_cols]

numeric_cols = df.select_dtypes(include=["number"]).columns
negative_counts = (df[numeric_cols] < 0).sum()

print("\n🔍 缺失值统计：\n", df.isnull().sum())
print("\n🔍 数值列负值统计：\n", negative_counts)

if negative_counts.sum() > 0:
    print("\n⚠️ 警告：存在负值，建议处理后再继续！")
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    print("✅ 已将负值替换为0。")

os.makedirs(os.path.dirname(FILTERED_DATA_CSV), exist_ok=True)
df.to_csv(FILTERED_DATA_CSV, index=False)
print(f"\n✅ 数据清洗完成并保存至：{FILTERED_DATA_CSV}")
