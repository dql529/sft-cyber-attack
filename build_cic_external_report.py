from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VAL_ORDER = {
    "E1_clean": 0,
    "E3_mask_30": 1,
    "E3_mask_50": 2,
    "E3_block_30": 3,
    "E3_block_50": 4,
    "E4_noise_30": 5,
    "E4_noise_50": 6,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-summary", default="./runs/cic_external_text/summary__medium.csv")
    ap.add_argument("--tabular-summary", default="./runs/cic_external_tabular/baselines_seed_summary_medium.csv")
    ap.add_argument("--outdir", default="./runs/cic_external_report")
    return ap.parse_args()


def aggregate_text(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["method_name"] == "tfidf_svm") & (df["text_col"] == "text")].copy()
    grouped = (
        df.groupby(["method_name", "val_experiment"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            seeds=("seed", "nunique"),
            total_params=("total_params", "mean"),
            trainable_params=("trainable_params", "mean"),
        )
        .fillna(0.0)
    )
    base = float(grouped[grouped["val_experiment"] == "E1_clean"]["macro_f1_mean"].iloc[0])
    grouped["delta_macro_f1_mean"] = grouped["macro_f1_mean"] - base
    grouped["delta_macro_f1_std"] = 0.0
    grouped["method_family"] = "text_baseline"
    grouped["text_col"] = "text"
    return grouped


def aggregate_tabular(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method_name"].isin(["MLP", "MLP_MaskAug"])].copy()
    df["method_family"] = "tabular"
    df["text_col"] = "structured_features"
    df["seeds"] = 3
    return df[
        [
            "method_family",
            "method_name",
            "text_col",
            "val_experiment",
            "acc_mean",
            "acc_std",
            "macro_f1_mean",
            "macro_f1_std",
            "delta_macro_f1_mean",
            "delta_macro_f1_std",
            "seeds",
        ]
    ].copy()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text_df = aggregate_text(Path(args.text_summary))
    tab_df = aggregate_tabular(Path(args.tabular_summary))
    merged = pd.concat([text_df, tab_df], ignore_index=True, sort=False)
    merged["val_rank"] = merged["val_experiment"].map(VAL_ORDER).fillna(999).astype(int)
    merged = merged.sort_values(["val_rank", "macro_f1_mean"], ascending=[True, False]).drop(columns=["val_rank"])
    merged.to_csv(outdir / "cic_external_table__medium.csv", index=False)

    missing_exps = [exp for exp in merged["val_experiment"].unique().tolist() if exp.startswith("E3_")]
    support = True
    for exp in missing_exps:
        sub = merged[merged["val_experiment"] == exp].set_index("method_name")
        if "MLP" not in sub.index or "MLP_MaskAug" not in sub.index:
            support = False
            break
        if float(sub.loc["MLP_MaskAug", "macro_f1_mean"]) <= float(sub.loc["MLP", "macro_f1_mean"]):
            support = False
            break
    manifest = {
        "dataset": "cicids2017_binary",
        "supports_missingness_direction": bool(support),
        "missingness_experiments": sorted(missing_exps),
    }
    with open(outdir / "cic_external_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[CIC REPORT] saved -> {outdir}")


if __name__ == "__main__":
    main()
