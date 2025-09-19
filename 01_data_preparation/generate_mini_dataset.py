import pandas as pd
import os

# --- 1. 配置 ---
# 输入文件路径
INPUT_CSV_PATH = "./data/prompt_csv/unsw15_prompt_train.csv"
# 输出文件路径
OUTPUT_CSV_PATH = "./data/prompt_csv/unsw15_prompt_train_mini.csv"

# 每个类别要抽取的样本数量
SAMPLES_PER_CLASS = 300
# 标签列的名称
LABEL_COLUMN = "label"
# 随机种子，确保每次运行结果一致
RANDOM_STATE = 42

# --- 2. 主程序 ---
print("--- 开始创建迷你数据集 ---")

# 检查输入文件是否存在
if not os.path.exists(INPUT_CSV_PATH):
    print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'")
else:
    # 读取原始数据集
    print(f"正在读取原始数据集: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    print("\n原始数据集信息:")
    print(f" - 形状: {df.shape}")
    print(f" - 各类别数量:\n{df[LABEL_COLUMN].value_counts().to_string()}")

    # 定义一个安全的采样函数，以防某个类别样本数不足100
    def safe_sample(group):
        if len(group) < SAMPLES_PER_CLASS:
            print(
                f"警告: 类别 '{group.name}' 样本数不足 {SAMPLES_PER_CLASS}，将使用所有 {len(group)} 个样本。"
            )
            return group
        return group.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_STATE)

    # 按 'label' 分组，并对每个组应用采样函数
    print(f"\n正在为每个类别采样 {SAMPLES_PER_CLASS} 个样本...")
    mini_df = df.groupby(LABEL_COLUMN, group_keys=False).apply(safe_sample)

    # 将生成的数据集随机打乱顺序
    print("正在打乱新数据集的顺序...")
    mini_df = mini_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 保存新的迷你数据集
    print(f"正在保存迷你数据集到: {OUTPUT_CSV_PATH}")
    mini_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n--- 操作成功 ---")
    print("最终迷你数据集信息:")
    print(f" - 形状: {mini_df.shape}")
    print(f" - 各类别数量:\n{mini_df[LABEL_COLUMN].value_counts().to_string()}")
    print("--------------------")
