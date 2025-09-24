# nn_tabular_baseline.py —— 结构化特征 + MLP（不依赖BERT）
import os, sys, time, random, math
import numpy as np, pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 路径工具 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))


def rp(p: str) -> str:
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(ROOT, p.lstrip("./"))
    return p


# ---------- 导入配置 ----------
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import FILTERED_DATA_CSV, LABEL_MAP

# ========== 顶部开关 ==========
DATA_TIER = "mini"  # "mini" | "medium" | "full"
RANDOM_STATE = 11
TEST_SIZE = 0.2  # 8:2 切分
PER_CLASS_CAP = {"mini": 300, "medium": 3000}  # 每类上限；full 不限

# 模型/训练参数
HIDDEN_SIZES = [512, 256]
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
EPOCHS = 30
PATIENCE = 6  # 早停耐心
NUM_WORKERS = 0  # Windows 上用0最稳
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = rp("./nn_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
# =================================


# ---------- 固定随机种子 ----------
def set_seed(s=11):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(RANDOM_STATE)

# ---------- 读取数据 ----------
csv_path = rp(FILTERED_DATA_CSV)
df = pd.read_csv(csv_path)
print(f"[I/O] loaded {len(df)} rows from {csv_path}")

# ---------- 标签映射 ----------
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

# ---------- 选择特征 ----------
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

# 附加自动识别（谨慎）
for c in df.columns:
    if c in cat_cols or c in num_cols or c in ["attack_cat", "label"]:
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        num_cols.append(c)
    elif pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object":
        cat_cols.append(c)

# 去重
cat_cols = list(dict.fromkeys(cat_cols))
num_cols = list(dict.fromkeys(num_cols))

if not (cat_cols or num_cols):
    raise RuntimeError("No usable feature columns detected.")

print(
    f"[FEAT] numeric({len(num_cols)}): {num_cols[:15]}{'...' if len(num_cols)>15 else ''}"
)
print(
    f"[FEAT] categoric({len(cat_cols)}): {cat_cols[:15]}{'...' if len(cat_cols)>15 else ''}"
)

# ---------- 按 tier 分层抽样 ----------
if DATA_TIER in PER_CLASS_CAP:
    cap = PER_CLASS_CAP[DATA_TIER]
    dfs = []
    for y, g in df.groupby("label", as_index=False):
        if len(g) > cap:
            dfs.append(g.sample(n=cap, random_state=RANDOM_STATE))
        else:
            dfs.append(g)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[SAMPLE] tier={DATA_TIER}, per-class cap={cap} -> {len(df)} rows")
else:
    print(f"[SAMPLE] tier=full, using all {len(df)} rows")

# ---------- 切分 ----------
X = df[cat_cols + num_cols].copy()
y = df["label"].to_numpy()
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[SPLIT] train={len(X_tr)}, val={len(X_va)}")

# ---------- 预处理（OneHot + 标准化） ----------
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(with_mean=True), num_cols),
    ],
    remainder="drop",
)
Xtr = ct.fit_transform(X_tr).astype(np.float32)
Xva = ct.transform(X_va).astype(np.float32)
in_dim = Xtr.shape[1]
n_classes = len(LABEL_MAP)
print(f"[SHAPE] Xtr={Xtr.shape}, Xva={Xva.shape}, classes={n_classes}")


# ---------- Dataset ----------
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


ds_tr = NumpyDataset(Xtr, y_tr)
ds_va = NumpyDataset(Xva, y_va)
dl_tr = DataLoader(
    ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
dl_va = DataLoader(
    ds_va,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ---------- MLP ----------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=[512, 256], pdrop=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(pdrop)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(in_dim, n_classes, hidden=HIDDEN_SIZES, pdrop=DROPOUT).to(DEVICE)

# ---------- 类别权重 ----------
cnt = np.bincount(y_tr, minlength=n_classes)
cw = (len(y_tr) / (n_classes * np.clip(cnt, 1, None))).astype(np.float32)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---------- 评估函数 ----------
from sklearn.metrics import f1_score


def eval_epoch():
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)
            all_y.append(yb.cpu().numpy())
            all_p.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    macro = f1_score(
        y_true, y_pred, labels=list(LABEL_MAP.keys()), average="macro", zero_division=0
    )
    acc = (y_true == y_pred).mean()
    return acc, macro, y_true, y_pred


# ---------- 训练 ----------
best_macro = -1.0
best_state = None
epochs_no_improve = 0

print(f"[TRAIN] device={DEVICE}, epochs={EPOCHS}, batch={BATCH_SIZE}")
for ep in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for xb, yb in dl_tr:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        ep_loss += loss.item() * xb.size(0)
    ep_loss /= len(ds_tr)

    acc, macro, _, _ = eval_epoch()
    print(
        f"  epoch {ep:02d} | train_loss={ep_loss:.4f} | val_acc={acc:.4f} | val_macroF1={macro:.4f}"
    )

    if macro > best_macro + 1e-4:
        best_macro = macro
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"[EARLY STOP] no improve for {PATIENCE} epochs.")
            break

# ---------- 最终评估 ----------
if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
acc, macro, y_true, y_pred = eval_epoch()

print("\n=== Tabular-MLP Report ===")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys())),
)
print(f"[RESULT] val_acc={acc:.4f}  val_macroF1={macro:.4f}")

# ---------- 保存 ----------
save_pt = os.path.join(OUT_DIR, f"tabmlp_{DATA_TIER}_best.pt")
torch.save(
    {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "classes": LABEL_MAP,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "ohe_scaler": "sklearn-ct-fit-in-script-not-serialized",
    },
    save_pt,
)
print("[SAVE]", save_pt)
