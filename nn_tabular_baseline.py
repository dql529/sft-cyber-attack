"""
Structured-feature baselines for the robustness study.

Models:
- MLP
- CNN1D
- Linear SVM
- Logistic Regression

The script trains on a clean split once per seed and evaluates on multiple
validation perturbations with unified metrics and runtime artifacts.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from robustness_utils import (
    DEFAULT_VAL_EXPERIMENTS,
    compute_classification_metrics,
    count_torch_parameters,
    ensure_dir,
    estimate_linear_probe_params,
    flatten_metric_record,
    mean_latency_ms,
    parse_csv_list,
    parse_int_list,
    peak_memory_mb,
    reset_peak_memory,
    rp,
    save_json,
    set_seed,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import FILTERED_DATA_CSV, LABEL_MAP


HIDDEN_SIZES = [512, 256]
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PER_CLASS_CAP = {"mini": 300, "medium": 3000}
LEGACY_CAT_COLS = ["proto", "state", "service"]
LEGACY_NUM_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sload", "dload", "tcprtt", "ct_srv_src"]


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=HIDDEN_SIZES, pdrop=DROPOUT):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(pdrop)])
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_dim, n_classes, pdrop=DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(pdrop), nn.Linear(128, n_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = self.conv(x).squeeze(-1)
        return self.head(feat)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--seeds", default="11,21,31")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--dataset-csv", default=FILTERED_DATA_CSV)
    ap.add_argument("--train-csv-override", default="")
    ap.add_argument("--val-csv-override", default="")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--label-name-col", default="label_name")
    ap.add_argument(
        "--metadata-cols",
        default="dataset_name,source_row_id,source_file,source_label,label_name,label,text",
    )
    ap.add_argument("--feature-mode", choices=["available", "legacy_selected"], default="available")
    ap.add_argument(
        "--models",
        default="MLP,MLP_MaskAug,CNN1D,Linear SVM,Logistic Regression",
        help="Comma-separated model list.",
    )
    ap.add_argument("--train-mask-ratios", default="0.1,0.3")
    ap.add_argument("--train-block-ratios", default="0.1")
    ap.add_argument("--train-noise-ratios", default="")
    ap.add_argument("--maskaug-label", default="MLP_MaskAug")
    ap.add_argument("--split-index", default="./data/prompt_csv/splits_unsw15_seed{seed}.npz")
    ap.add_argument("--val-experiments", default=",".join(DEFAULT_VAL_EXPERIMENTS))
    ap.add_argument("--output-root", default="./nn_outputs")
    return ap.parse_args()


def apply_missing_mask(df: pd.DataFrame, cols, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in cols:
        mask = rng.random(len(out)) < ratio
        out.loc[mask, col] = np.nan
    return out


def apply_noise(df: pd.DataFrame, cols, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in cols:
        values = out[col].to_numpy()
        mask = np.isfinite(values) & (values > 0)
        values = values.astype(float, copy=True)
        values[mask] = values[mask] * (1.0 + rng.uniform(-ratio, ratio, size=mask.sum()))
        out[col] = values
    return out


def apply_block_missing_mask(df: pd.DataFrame, blocks, ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for block in blocks:
        present = [col for col in block if col in out.columns]
        if not present:
            continue
        mask = rng.random(len(out)) < ratio
        out.loc[mask, present] = np.nan
    return out


def stratified_cap(df: pd.DataFrame, cap: int, seed: int):
    if cap <= 0:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("label", as_index=False):
        if len(group) > cap:
            parts.append(group.sample(n=cap, random_state=int(rng.integers(0, 1e9))))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def parse_ratio_list(value: str):
    if not value:
        return []
    parts = []
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "na", "n/a"}:
            continue
        parts.append(float(token))
    return parts


def infer_feature_columns(df: pd.DataFrame, feature_mode: str, extra_exclude=None):
    exclude = {"attack_cat", "label", "id", "source_row_id", "label_name", "dataset_name"}
    if extra_exclude:
        exclude.update({str(x).strip() for x in extra_exclude if str(x).strip()})
    if feature_mode == "legacy_selected":
        cat_cols = [c for c in LEGACY_CAT_COLS if c in df.columns]
        num_cols = [c for c in LEGACY_NUM_COLS if c in df.columns]
        return cat_cols, num_cols

    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = []
    num_cols = []
    for col in feature_cols:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


def infer_block_groups(num_cols):
    known_groups = [
        ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"],
        ["sttl", "dttl", "sload", "dload", "sloss", "dloss"],
        ["sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"],
        ["swin", "stcpb", "dtcpb", "dwin", "smean", "dmean"],
        [
            "trans_depth",
            "response_body_len",
            "ct_srv_src",
            "ct_state_ttl",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
            "ct_src_ltm",
            "ct_srv_dst",
            "is_ftp_login",
            "is_sm_ips_ports",
        ],
    ]
    groups = []
    assigned = set()
    for group in known_groups:
        present = [col for col in group if col in num_cols]
        if present:
            groups.append(present)
            assigned.update(present)
    leftover = [col for col in num_cols if col not in assigned]
    if leftover:
        step = 4
        for start in range(0, len(leftover), step):
            groups.append(leftover[start : start + step])
    return groups


def infer_label_schema_from_frames(frames, label_col: str, label_name_col: str):
    raw_ids = sorted(
        {
            int(v)
            for df in frames
            if label_col in df.columns
            for v in df[label_col].dropna().astype(int).tolist()
        }
    )
    raw_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_ids)}
    label_name_map = {}
    for df in frames:
        if {label_col, label_name_col}.issubset(df.columns):
            pairs = df[[label_col, label_name_col]].dropna().drop_duplicates()
            for raw_id, name in pairs.itertuples(index=False):
                label_name_map[int(raw_id)] = str(name)
    label_names = [label_name_map.get(raw_id, LABEL_MAP.get(raw_id, f"class_{raw_id}")) for raw_id in raw_ids]
    return raw_ids, raw_to_idx, label_names


def build_transformer(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="UNK"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                cat_cols,
            ),
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
                    StandardScaler(with_mean=True),
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )


def build_val_variants(val_df: pd.DataFrame, num_cols, experiments: list[str], split_seed: int):
    out = {"E1_clean": val_df.copy()}
    block_groups = infer_block_groups(num_cols)
    for exp in experiments:
        if exp == "E1_clean":
            continue
        if exp.startswith("E3_mask_"):
            ratio = float(exp.split("_")[-1]) / 100.0
            out[exp] = apply_missing_mask(val_df, num_cols, ratio, seed=split_seed + 100)
        elif exp.startswith("E3_block_"):
            ratio = float(exp.split("_")[-1]) / 100.0
            out[exp] = apply_block_missing_mask(val_df, block_groups, ratio, seed=split_seed + 150)
        elif exp.startswith("E4_noise_"):
            ratio = float(exp.split("_")[-1]) / 100.0
            out[exp] = apply_noise(val_df, num_cols, ratio, seed=split_seed + 200)
        else:
            raise ValueError(f"Unsupported validation experiment '{exp}'.")
    return out


def build_augmented_train_df(train_df, num_cols, split_seed: int, mask_ratios, block_ratios, noise_ratios):
    frames = [train_df.copy()]
    block_groups = infer_block_groups(num_cols)
    for ratio in mask_ratios:
        frames.append(apply_missing_mask(train_df, num_cols, ratio, seed=split_seed + 1000 + int(ratio * 100)))
    for ratio in block_ratios:
        frames.append(
            apply_block_missing_mask(train_df, block_groups, ratio, seed=split_seed + 1100 + int(ratio * 100))
        )
    for ratio in noise_ratios:
        frames.append(apply_noise(train_df, num_cols, ratio, seed=split_seed + 1200 + int(ratio * 100)))
    return pd.concat(frames, ignore_index=True)


def load_data(
    tier: str,
    split_seed: int,
    split_index: str,
    dataset_csv: str,
    feature_mode: str,
    train_csv_override: str = "",
    val_csv_override: str = "",
    label_col: str = "label",
    label_name_col: str = "label_name",
    metadata_cols=None,
):
    metadata_cols = metadata_cols or []
    if train_csv_override or val_csv_override:
        if not train_csv_override or not val_csv_override:
            raise RuntimeError("Both --train-csv-override and --val-csv-override are required together.")

        train_csv = rp(train_csv_override)
        val_csv = rp(val_csv_override)
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        for csv_path, df in [(train_csv, train_df), (val_csv, val_df)]:
            if label_col not in df.columns:
                raise RuntimeError(f"{csv_path} is missing label column '{label_col}'.")
        raw_ids, raw_to_idx, label_names = infer_label_schema_from_frames(
            [train_df, val_df], label_col=label_col, label_name_col=label_name_col
        )
        if not raw_ids:
            raise RuntimeError("No labels found in override CSVs.")
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["label"] = train_df[label_col].astype(int).map(raw_to_idx)
        val_df["label"] = val_df[label_col].astype(int).map(raw_to_idx)
        train_df = train_df.dropna(subset=["label"]).copy()
        val_df = val_df.dropna(subset=["label"]).copy()
        train_df["label"] = train_df["label"].astype(int)
        val_df["label"] = val_df["label"].astype(int)

        if tier in PER_CLASS_CAP:
            train_df = stratified_cap(train_df, PER_CLASS_CAP[tier], split_seed)
            val_df = stratified_cap(val_df, PER_CLASS_CAP[tier], split_seed + 1)

        extra_exclude = set(metadata_cols)
        extra_exclude.add(label_col)
        extra_exclude.add(label_name_col)
        cat_cols, num_cols = infer_feature_columns(train_df, feature_mode, extra_exclude=extra_exclude)
        feature_cols = cat_cols + num_cols
        missing_train = set(feature_cols) - set(train_df.columns)
        missing_val = set(feature_cols) - set(val_df.columns)
        if missing_train or missing_val:
            raise RuntimeError(
                f"Override CSVs are missing feature columns. train={sorted(missing_train)} val={sorted(missing_val)}"
            )
        if num_cols:
            train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
            val_df[num_cols] = val_df[num_cols].replace([np.inf, -np.inf], np.nan)

        transformer = build_transformer(cat_cols, num_cols)
        X_tr = transformer.fit_transform(train_df[feature_cols]).astype(np.float32)
        y_tr = train_df["label"].to_numpy()
        return train_df, val_df, transformer, cat_cols, num_cols, feature_cols, X_tr, y_tr, list(range(len(raw_ids))), label_names

    csv_path = rp(dataset_csv)
    df = pd.read_csv(csv_path)
    label2id = {v: k for k, v in LABEL_MAP.items()}
    if "attack_cat" not in df.columns:
        raise RuntimeError("Column 'attack_cat' not found in FILTERED_DATA_CSV.")
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df["label"] = df["attack_cat"].map(label2id)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    raw_ids, raw_to_idx, label_names = infer_label_schema_from_frames(
        [df.assign(label_name=df["attack_cat"])], label_col="label", label_name_col="label_name"
    )
    df["label"] = df["label"].map(raw_to_idx).astype(int)

    cat_cols, num_cols = infer_feature_columns(df, feature_mode)
    required = set(cat_cols + num_cols + ["attack_cat", "label"])
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    split_path = split_index.format(seed=split_seed) if "{seed}" in split_index else split_index
    split_path = rp(split_path)
    if not os.path.exists(split_path):
        raise RuntimeError(f"Split index not found: {split_path}. Run prepare.py first.")

    idx = np.load(split_path)
    train_df = df.iloc[idx["train_idx"]].copy()
    val_df = df.iloc[idx["val_idx"]].copy()

    if tier in PER_CLASS_CAP:
        train_df = stratified_cap(train_df, PER_CLASS_CAP[tier], split_seed)
        val_df = stratified_cap(val_df, PER_CLASS_CAP[tier], split_seed + 1)

    feature_cols = cat_cols + num_cols
    if num_cols:
        train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
        val_df[num_cols] = val_df[num_cols].replace([np.inf, -np.inf], np.nan)
    transformer = build_transformer(cat_cols, num_cols)

    X_tr = transformer.fit_transform(train_df[feature_cols]).astype(np.float32)
    y_tr = train_df["label"].to_numpy()
    return train_df, val_df, transformer, cat_cols, num_cols, feature_cols, X_tr, y_tr, list(range(len(raw_ids))), label_names


def eval_torch_model(model, dl):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            pred = model(xb).argmax(1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def train_torch_model(model, X_tr, y_tr, X_val_clean, y_val_clean, label_ids, label_names):
    ds_tr = NumpyDataset(X_tr, y_tr)
    ds_va = NumpyDataset(X_val_clean, y_val_clean)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    n_classes = len(label_ids)
    counts = np.bincount(y_tr, minlength=n_classes)
    class_weights = (len(y_tr) / (n_classes * np.clip(counts, 1, None))).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=DEVICE))

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_macro = -1.0
    best_state = None
    no_improve = 0

    reset_peak_memory()
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        val_pred = eval_torch_model(model, dl_va)
        metrics = compute_classification_metrics(
            y_val_clean,
            val_pred,
            label_ids=label_ids,
            label_names=label_names,
        )
        macro = metrics["macro_f1"]
        print(f"  epoch {epoch:02d} | clean_val_macro_f1={macro:.4f}")
        if macro > best_macro + 1e-4:
            best_macro = macro
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    train_s = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, train_s, peak_memory_mb()


def predict_torch(model, X):
    ds = NumpyDataset(X, np.zeros(len(X), dtype=np.int64))
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    start = time.perf_counter()
    y_pred = eval_torch_model(model, dl)
    predict_s = time.perf_counter() - start
    return y_pred, predict_s


def sklearn_param_count(pipe):
    return estimate_linear_probe_params(pipe)


def save_metric_artifact(output_root, payload):
    method = payload["method_name"].replace(" ", "_").lower()
    exp = payload["val_experiment"]
    seed = payload["seed"]
    metrics_path = os.path.join(output_root, f"metrics__{method}__seed-{seed}__{exp}.json")
    save_json(metrics_path, payload)
    return metrics_path


def main():
    args = parse_args()
    ensure_dir(rp(args.output_root))
    experiments = parse_csv_list(args.val_experiments)
    seeds = parse_int_list(args.seeds)
    models = parse_csv_list(args.models)
    train_mask_ratios = parse_ratio_list(args.train_mask_ratios)
    train_block_ratios = parse_ratio_list(args.train_block_ratios)
    train_noise_ratios = parse_ratio_list(args.train_noise_ratios)
    metadata_cols = parse_csv_list(args.metadata_cols)
    train_df, val_df, transformer, cat_cols, num_cols, feature_cols, X_tr, y_tr, label_ids, label_names = load_data(
        tier=args.tier,
        split_seed=args.split_seed,
        split_index=args.split_index,
        dataset_csv=args.dataset_csv,
        feature_mode=args.feature_mode,
        train_csv_override=args.train_csv_override,
        val_csv_override=args.val_csv_override,
        label_col=args.label_col,
        label_name_col=args.label_name_col,
        metadata_cols=metadata_cols,
    )
    val_variants = build_val_variants(val_df, num_cols, experiments, args.split_seed)
    train_aug_df = build_augmented_train_df(
        train_df,
        num_cols,
        args.split_seed,
        train_mask_ratios,
        train_block_ratios,
        train_noise_ratios,
    )
    X_tr_aug = transformer.transform(train_aug_df[feature_cols]).astype(np.float32)
    y_tr_aug = train_aug_df["label"].to_numpy()

    rows = []
    for seed in seeds:
        set_seed(seed)
        X_val_clean = transformer.transform(val_variants["E1_clean"][feature_cols]).astype(np.float32)
        y_val_clean = val_variants["E1_clean"]["label"].to_numpy()

        print(f"[RUN] tabular seed={seed} tier={args.tier}")
        model_specs = []
        if "MLP" in models:
            mlp, mlp_train_s, mlp_peak_mem = train_torch_model(
                MLP(X_tr.shape[1], len(label_ids)),
                X_tr,
                y_tr,
                X_val_clean,
                y_val_clean,
                label_ids,
                label_names,
            )
            torch.save(mlp.state_dict(), os.path.join(rp(args.output_root), f"mlp_{args.tier}_seed{seed}.pt"))
            model_specs.append(
                (
                    "MLP",
                    count_torch_parameters(mlp),
                    mlp_train_s,
                    mlp_peak_mem,
                    lambda arr: predict_torch(mlp, arr),
                )
            )
        if "MLP_MaskAug" in models:
            mlp_aug, mlp_aug_train_s, mlp_aug_peak_mem = train_torch_model(
                MLP(X_tr_aug.shape[1], len(label_ids)),
                X_tr_aug,
                y_tr_aug,
                X_val_clean,
                y_val_clean,
                label_ids,
                label_names,
            )
            torch.save(
                mlp_aug.state_dict(),
                os.path.join(
                    rp(args.output_root),
                    f"{args.maskaug_label.replace(' ', '_').lower()}_{args.tier}_seed{seed}.pt",
                ),
            )
            model_specs.append(
                (
                    args.maskaug_label,
                    count_torch_parameters(mlp_aug),
                    mlp_aug_train_s,
                    mlp_aug_peak_mem,
                    lambda arr: predict_torch(mlp_aug, arr),
                )
            )
        if "CNN1D" in models:
            cnn, cnn_train_s, cnn_peak_mem = train_torch_model(
                CNN1D(X_tr.shape[1], len(label_ids)),
                X_tr,
                y_tr,
                X_val_clean,
                y_val_clean,
                label_ids,
                label_names,
            )
            torch.save(cnn.state_dict(), os.path.join(rp(args.output_root), f"cnn1d_{args.tier}_seed{seed}.pt"))
            model_specs.append(
                (
                    "CNN1D",
                    count_torch_parameters(cnn),
                    cnn_train_s,
                    cnn_peak_mem,
                    lambda arr: predict_torch(cnn, arr),
                )
            )
        if "Linear SVM" in models:
            svm_start = time.perf_counter()
            svm = make_pipeline(
                StandardScaler(with_mean=True),
                LinearSVC(class_weight="balanced", random_state=seed),
            )
            svm.fit(X_tr, y_tr)
            svm_train_s = time.perf_counter() - svm_start
            model_specs.append(
                (
                    "Linear SVM",
                    {
                        "total_params": sklearn_param_count(svm),
                        "trainable_params": sklearn_param_count(svm),
                    },
                    svm_train_s,
                    0.0,
                    lambda arr: (svm.predict(arr), 0.0),
                )
            )
        if "Logistic Regression" in models:
            lr_start = time.perf_counter()
            logreg = make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            logreg.fit(X_tr, y_tr)
            lr_train_s = time.perf_counter() - lr_start
            model_specs.append(
                (
                    "Logistic Regression",
                    {
                        "total_params": sklearn_param_count(logreg),
                        "trainable_params": sklearn_param_count(logreg),
                    },
                    lr_train_s,
                    0.0,
                    lambda arr: (logreg.predict(arr), 0.0),
                )
            )

        for model_name, params, train_s, peak_mem, predict_fn in model_specs:
            for exp_name, df_variant in val_variants.items():
                X_va = transformer.transform(df_variant[feature_cols]).astype(np.float32)
                y_true = df_variant["label"].to_numpy()
                if model_name in {"Linear SVM", "Logistic Regression"}:
                    pred_start = time.perf_counter()
                    y_pred = predict_fn(X_va)[0]
                    predict_s = time.perf_counter() - pred_start
                else:
                    y_pred, predict_s = predict_fn(X_va)
                metrics = compute_classification_metrics(
                    y_true,
                    y_pred,
                    label_ids=label_ids,
                    label_names=label_names,
                )
                runtime = {
                    "train_s": float(train_s),
                    "predict_s": float(predict_s),
                    "inference_latency_ms_per_sample": mean_latency_ms(predict_s, len(y_true)),
                    "peak_gpu_mem_mb": float(peak_mem),
                }
                row = flatten_metric_record(
                    method_family="tabular",
                    method_name=model_name,
                    tier=args.tier,
                    text_col="structured_features",
                    train_experiment="E1_clean",
                    val_experiment=exp_name,
                    seed=seed,
                    split_seed=args.split_seed,
                    metrics=metrics,
                    runtime=runtime,
                    params=params,
                    extra={
                        "dataset_csv": rp(args.dataset_csv),
                        "train_csv_override": rp(args.train_csv_override) if args.train_csv_override else "",
                        "val_csv_override": rp(args.val_csv_override) if args.val_csv_override else "",
                        "feature_mode": args.feature_mode,
                        "n_cat_features": len(cat_cols),
                        "n_num_features": len(num_cols),
                        "train_rows_clean": int(len(train_df)),
                        "train_rows_effective": int(len(train_aug_df))
                        if model_name == args.maskaug_label
                        else int(len(train_df)),
                    },
                )
                artifact = {
                    **row,
                    "metrics": metrics,
                    "runtime": runtime,
                    "params": params,
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(df_variant)),
                    "dataset_csv": rp(args.dataset_csv),
                    "train_csv_override": rp(args.train_csv_override) if args.train_csv_override else "",
                    "val_csv_override": rp(args.val_csv_override) if args.val_csv_override else "",
                    "feature_mode": args.feature_mode,
                    "cat_cols": cat_cols,
                    "num_cols": num_cols,
                }
                metrics_path = save_metric_artifact(rp(args.output_root), artifact)
                row["metrics_json"] = os.path.basename(metrics_path)
                rows.append(row)

    df = pd.DataFrame(rows)
    base = (
        df[df["val_experiment"] == "E1_clean"][["method_name", "seed", "split_seed", "macro_f1"]]
        .rename(columns={"macro_f1": "macro_f1_base"})
    )
    df = df.merge(base, on=["method_name", "seed", "split_seed"], how="left")
    df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]

    summary_path = os.path.join(rp(args.output_root), f"baselines_metrics_{args.tier}.csv")
    df.to_csv(summary_path, index=False)
    agg = (
        df.groupby(["method_name", "val_experiment"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_std=("delta_macro_f1", "std"),
        )
        .fillna(0.0)
    )
    agg_path = os.path.join(rp(args.output_root), f"baselines_seed_summary_{args.tier}.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[METRICS] saved -> {summary_path}")
    print(f"[SEED SUMMARY] saved -> {agg_path}")


if __name__ == "__main__":
    main()
