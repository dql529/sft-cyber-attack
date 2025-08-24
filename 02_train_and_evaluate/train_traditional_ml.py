# train_traditional_ml.py

import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILTERED_DATA_CSV

print("开始使用传统机器学习方法（随机森林）进行训练和评估...")

# 1. 加载数据
df = pd.read_csv(FILTERED_DATA_CSV)

# 确保 'attack_cat' 列存在
if 'attack_cat' not in df.columns:
    raise ValueError("错误: 'attack_cat' 列在数据集中未找到！")

# 2. 定义特征 (X) 和目标 (y)
X = df.drop("attack_cat", axis=1)
y = df["attack_cat"]

# 3. 识别不同类型的特征
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\n识别到 {len(numeric_features)} 个数值特征: {numeric_features}")
print(f"识别到 {len(categorical_features)} 个类别特征: {categorical_features}")

# 4. 创建预处理流程
# 对数值特征进行标准化，对类别特征进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

# 5. 定义模型
# 使用随机森林分类器，n_jobs=-1表示使用所有可用的CPU核心
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 6. 创建完整的机器学习工作流 (Pipeline)
# 将预处理步骤和模型训练步骤串联起来
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# 7. 划分训练集和测试集
# stratify=y 确保在划分时，训练集和测试集中的类别分布与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n数据集划分为 {len(X_train)} 条训练样本和 {len(X_test)} 条测试样本。")

# 8. 训练模型
print("\n正在训练随机森林模型...")
pipeline.fit(X_train, y_train)
print("模型训练完成！")

# 9. 在测试集上进行评估
print("\n正在测试集上进行评估...")
y_pred = pipeline.predict(X_test)

# 10. 打印评估报告
print("\n" + "=" * 30)
print("传统方法 (随机森林) 评估报告:")
print("=" * 30)
print(classification_report(y_test, y_pred))