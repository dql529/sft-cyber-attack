import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from robustness_utils import DEFAULT_VAL_EXPERIMENTS, ensure_dir, parse_experiment_metadata, rp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--probe-summary", default="./probe_outputs_any/summary__medium.csv")
    ap.add_argument("--tabular-summary", default="./nn_outputs/baselines_metrics_medium.csv")
    ap.add_argument("--text-summary", default="./text_baseline_outputs/summary__medium.csv")
    ap.add_argument(
        "--finetune-summary",
        default="./checkpoints/bert_classifier_robustness/summary__medium_text.csv",
    )
    ap.add_argument("--madfuse-summary", default="./madfuse_outputs/summary__medium.csv")
    ap.add_argument("--probe-root", default="./probe_outputs_any")
    ap.add_argument("--tabular-root", default="./nn_outputs")
    ap.add_argument("--text-root", default="./text_baseline_outputs")
    ap.add_argument("--finetune-root", default="./checkpoints/bert_classifier_robustness")
    ap.add_argument("--madfuse-root", default="./madfuse_outputs")
    ap.add_argument("--output-dir", default="./robustness_report")
    return ap.parse_args()


def load_summary(path: str, family_hint: str) -> pd.DataFrame:
    path = rp(path)
    if not os.path.exists(path):
        print(f"[WARN] summary not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method_family" not in df.columns:
        df["method_family"] = family_hint
    if "text_col" not in df.columns:
        df["text_col"] = "structured_features" if family_hint == "tabular" else "text"
    return df


def method_label(row) -> str:
    text_col = row.get("text_col", "")
    if text_col in {"", "text", "structured_features"}:
        return row["method_name"]
    return f"{row['method_name']} ({text_col})"


def aggregate_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(cols, dropna=False, as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_std=("delta_macro_f1", "std"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            predict_s=("predict_s", "mean"),
            inference_latency_ms_per_sample=("inference_latency_ms_per_sample", "mean"),
            peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
        )
        .fillna(0.0)
    )
    grouped["method_label"] = grouped.apply(method_label, axis=1)
    return grouped


def load_metric_payloads(metric_roots):
    payloads = []
    for root in metric_roots:
        root = rp(root)
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.startswith("metrics__") or not fn.endswith(".json"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "r", encoding="utf-8") as f:
                    payloads.append(json.load(f))
    return payloads


def payload_method_name(payload: dict):
    return payload.get("method_name") or payload.get("model") or payload.get("method")


def pick_best_method(clean_df: pd.DataFrame, family: str):
    sub = clean_df[clean_df["method_family"] == family]
    if sub.empty:
        return None
    agg = (
        sub.groupby(["method_family", "method_name", "text_col"], as_index=False)
        .agg(macro_f1_mean=("macro_f1", "mean"))
        .sort_values(by="macro_f1_mean", ascending=False)
    )
    return agg.iloc[0].to_dict()


def build_classwise_plots(df: pd.DataFrame, metric_payloads: list[dict], out_dir: str):
    clean_df = df[df["val_experiment"] == "E1_clean"]
    selected = []
    for family in ["madfuse", "frozen_probe", "finetune", "tabular"]:
        best = pick_best_method(clean_df, family)
        if best is not None:
            selected.append(best)

    metric_map = defaultdict(dict)
    for payload in metric_payloads:
        key = (
            payload.get("method_family"),
            payload_method_name(payload),
            payload.get("text_col", "text"),
            payload.get("seed"),
            payload.get("split_seed"),
            payload.get("val_experiment"),
        )
        metric_map[key] = payload

    for spec in selected:
        family = spec["method_family"]
        method = spec["method_name"]
        text_col = spec["text_col"]
        rows = clean_df[
            (clean_df["method_family"] == family)
            & (clean_df["method_name"] == method)
            & (clean_df["text_col"] == text_col)
        ].sort_values(by="macro_f1", ascending=False)
        if rows.empty:
            continue
        best_row = rows.iloc[0]
        seed = int(best_row["seed"])
        split_seed = int(best_row["split_seed"])
        frames = []
        for exp in ["E1_clean", "E3_mask_30", "E4_noise_30"]:
            payload = metric_map.get((family, method, text_col, seed, split_seed, exp))
            if not payload:
                continue
            per_class = payload["metrics"]["per_class_f1"]
            frames.append(
                pd.DataFrame({"class": list(per_class.keys()), "f1": list(per_class.values()), "exp": exp})
            )
        if not frames:
            continue

        plot_df = pd.concat(frames, ignore_index=True)
        pivot = plot_df.pivot(index="class", columns="exp", values="f1").fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1")
        label = method if text_col in {"text", "structured_features"} else f"{method} ({text_col})"
        ax.set_title(f"Per-class F1: {label}")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"classwise__{family}__{method}__{text_col}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()


def build_robustness_curves(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return
    families = ["madfuse", "frozen_probe", "finetune", "text_baseline", "tabular"]
    clean_df = df[df["val_experiment"] == "E1_clean"]
    selected = []
    for family in families:
        best = pick_best_method(clean_df, family)
        if best is not None:
            selected.append(best)
    if not selected:
        return

    curve_rows = []
    for spec in selected:
        subset = df[
            (df["method_family"] == spec["method_family"])
            & (df["method_name"] == spec["method_name"])
            & (df["text_col"] == spec["text_col"])
        ].copy()
        if subset.empty:
            continue
        meta = subset["val_experiment"].map(parse_experiment_metadata)
        subset["curve_family"] = [item["family"] for item in meta]
        subset["curve_rate"] = [item["rate"] for item in meta]
        subset["method_label"] = subset.apply(method_label, axis=1)
        curve_rows.append(subset)
    if not curve_rows:
        return

    curve_df = pd.concat(curve_rows, ignore_index=True)
    curve_specs = [
        ("mask", "curve__mask_missing_macro_f1.png", "IID Missingness"),
        ("block", "curve__block_missing_macro_f1.png", "Block Missingness"),
        ("noise", "curve__noise_macro_f1.png", "Multiplicative Noise"),
    ]
    for family_name, filename, title in curve_specs:
        subset = curve_df[curve_df["curve_family"] == family_name]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["method_label", "curve_rate"], as_index=False)
            .agg(macro_f1_mean=("macro_f1", "mean"))
            .sort_values(by=["method_label", "curve_rate"])
        )
        plt.figure(figsize=(7, 4))
        for method_name, method_df in grouped.groupby("method_label"):
            plt.plot(method_df["curve_rate"], method_df["macro_f1_mean"], marker="o", label=method_name)
        plt.xlabel("Corruption Rate (%)")
        plt.ylabel("Macro-F1")
        plt.ylim(0.0, 1.0)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=200)
        plt.close()


def main():
    args = parse_args()
    out_dir = rp(args.output_dir)
    ensure_dir(out_dir)

    frames = [
        load_summary(args.probe_summary, "frozen_probe"),
        load_summary(args.tabular_summary, "tabular"),
        load_summary(args.text_summary, "text_baseline"),
        load_summary(args.finetune_summary, "finetune"),
        load_summary(args.madfuse_summary, "madfuse"),
    ]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame()
    if df.empty:
        print("[WARN] no summaries found; nothing to report.")
        return

    if "delta_macro_f1" not in df.columns and "macro_f1_base" in df.columns:
        df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]
    elif "delta_macro_f1" not in df.columns:
        df["delta_macro_f1"] = 0.0

    df["method_label"] = df.apply(method_label, axis=1)
    raw_path = os.path.join(out_dir, f"all_results__{args.tier}.csv")
    df.to_csv(raw_path, index=False)

    clean_table = aggregate_summary(
        df[df["val_experiment"] == "E1_clean"],
        ["method_family", "method_name", "text_col"],
    ).sort_values(by=["macro_f1_mean", "acc_mean"], ascending=False)
    clean_path = os.path.join(out_dir, f"main_clean_table__{args.tier}.csv")
    clean_table.to_csv(clean_path, index=False)

    robustness_table = aggregate_summary(
        df[df["val_experiment"].isin(DEFAULT_VAL_EXPERIMENTS)],
        ["method_family", "method_name", "text_col", "val_experiment"],
    ).sort_values(by=["val_experiment", "macro_f1_mean"], ascending=[True, False])
    robust_path = os.path.join(out_dir, f"robustness_table__{args.tier}.csv")
    robustness_table.to_csv(robust_path, index=False)

    efficiency_table = clean_table[
        [
            "method_family",
            "method_name",
            "text_col",
            "method_label",
            "total_params",
            "trainable_params",
            "predict_s",
            "inference_latency_ms_per_sample",
            "peak_gpu_mem_mb",
        ]
    ].copy()
    eff_path = os.path.join(out_dir, f"efficiency_table__{args.tier}.csv")
    efficiency_table.to_csv(eff_path, index=False)

    stability_table = (
        df[df["val_experiment"] == "E1_clean"]
        .groupby(["method_family", "method_name", "text_col"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .fillna(0.0)
    )
    stability_table["method_label"] = stability_table.apply(method_label, axis=1)
    stability_path = os.path.join(out_dir, f"stability_table__{args.tier}.csv")
    stability_table.to_csv(stability_path, index=False)

    metric_payloads = load_metric_payloads(
        [args.probe_root, args.tabular_root, args.text_root, args.finetune_root, args.madfuse_root]
    )
    build_classwise_plots(df, metric_payloads, out_dir)
    build_robustness_curves(df, out_dir)

    print(f"[SAVE] {raw_path}")
    print(f"[SAVE] {clean_path}")
    print(f"[SAVE] {robust_path}")
    print(f"[SAVE] {eff_path}")
    print(f"[SAVE] {stability_path}")


if __name__ == "__main__":
    main()
