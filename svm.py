# svm_traditional_baseline.py —— 纯传统SVM基线（不依赖BERT），用结构化原始特征
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import OneHotEncoder

# --- 项目路径工具（与现有一致） ---
ROOT = os.path.dirname(os.path.abspath(__file__))


def rp(p: str) -> str:
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(ROOT, p.lstrip("./"))
    return p


# --- 导入配置 ---
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import FILTERED_DATA_CSV, LABEL_MAP  # 直接用你现有 config

# ===== 顶部开关 =====
DATA_TIER = "mini"  # "mini" | "medium" | "full"
MODEL_TYPE = "linear"  # "linear" | "rbf"
RANDOM_STATE = 11

# 每类上限（仅对 mini / medium 生效；full 不限）
PER_CLASS_CAP = {
    "mini": 300,  # 每类最多 300
    "medium": 3000,  # 每类最多 3000（机器资源吃紧可调小）
}

TEST_SIZE = 0.2  # 8:2 切分
# ====================

# --- 加载数据 ---
csv_path = rp(FILTERED_DATA_CSV)
df = pd.read_csv(csv_path)
print(f"[I/O] loaded {len(df)} rows from {csv_path}")

# --- 标签映射 ---
label2id = {v: k for k, v in LABEL_MAP.items()}
if "attack_cat" not in df.columns:
    raise RuntimeError("Column 'attack_cat' not found in FILTERED_DATA_CSV.")
df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
df["label"] = df["attack_cat"].map(label2id)
n_drop = df["label"].isna().sum()
if n_drop > 0:
    print(f"[WARN] drop {n_drop} rows with unknown attack_cat.")
    df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# --- 选择特征列（尽量稳健：自动识别 + 白名单优先） ---
# 你之前常用的关键列：
whitelist_cats = ["proto", "state", "service"]
whitelist_nums = [
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "dur",
    "sload",
    "dload",
    "tcprtt",
    "ct_srv_src",
]

existing_cols = set(df.columns)
cat_cols = [c for c in whitelist_cats if c in existing_cols]
num_cols = [c for c in whitelist_nums if c in existing_cols]

# 附加自动识别（防止漏掉数值列）
for c in df.columns:
    if c in (cat_cols + num_cols + ["attack_cat", "label"]):
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        num_cols.append(c)
    elif pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object":
        # 避免把 text/prompt 之类极长文本混进来，这里做一个长度过滤
        # （UNSW这种结构化表一般不会有长文本）
        cat_cols.append(c)

# 去重
cat_cols = list(dict.fromkeys(cat_cols))
num_cols = list(dict.fromkeys(num_cols))

if not num_cols and not cat_cols:
    raise RuntimeError("No usable feature columns detected.")

print(
    f"[FEAT] numeric: {len(num_cols)} cols -> {num_cols[:15]}{'...' if len(num_cols)>15 else ''}"
)
print(
    f"[FEAT] categoric: {len(cat_cols)} cols -> {cat_cols[:15]}{'...' if len(cat_cols)>15 else ''}"
)

# --- 分层抽样到 mini / medium ---
if DATA_TIER in PER_CLASS_CAP:
    cap = PER_CLASS_CAP[DATA_TIER]
    dfs = []
    for y, g in df.groupby("label", as_index=False):
        if len(g) > cap:
            dfs.append(g.sample(n=cap, random_state=RANDOM_STATE))
        else:
            dfs.append(g)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[SAMPLE] DATA_TIER={DATA_TIER}, per-class cap={cap} -> kept {len(df)} rows")
else:
    print(f"[SAMPLE] DATA_TIER=full, use all {len(df)} rows")

# --- 切分 train/val（分层） ---
X = df[cat_cols + num_cols].copy()
y = df["label"].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[SPLIT] train={len(X_train)}, val={len(X_val)}")

# --- 预处理 + 模型 ---
ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,  # 让 OHE 稀疏矩阵在一定比例下仍保留稀疏
)

if MODEL_TYPE == "linear":
    clf = LinearSVC(
        C=1.0, loss="squared_hinge", class_weight="balanced", random_state=RANDOM_STATE
    )
elif MODEL_TYPE == "rbf":
    # 警告：在 medium/full 上会很慢很占内存
    clf = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
else:
    raise ValueError("MODEL_TYPE must be 'linear' or 'rbf'.")

pipe = make_pipeline(ct, clf)
print(f"[FIT] MODEL_TYPE={MODEL_TYPE} ...")
t0 = time.time()
pipe.fit(X_train, y_train)
fit_s = time.time() - t0

# --- 评估 ---
y_pred = pipe.predict(X_val)
from sklearn.metrics import precision_recall_fscore_support

print("\n=== Traditional SVM Report (no BERT) ===")
print(
    classification_report(
        y_val,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(y_val, y_pred, labels=list(LABEL_MAP.keys())),
)
print(f"[TIME] fit_sec={fit_s:.2f}")
