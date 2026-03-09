from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VAL_ORDER = {
    "E1_clean": 0,
    "E3_mask_10": 1,
    "E3_mask_30": 2,
    "E3_mask_50": 3,
    "E3_block_10": 4,
    "E3_block_30": 5,
    "E3_block_50": 6,
    "E4_noise_10": 7,
    "E4_noise_30": 8,
    "E4_noise_50": 9,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--text-summary",
        default="./runs/lightweight_text_medium_3seed/summary__medium.csv",
    )
    ap.add_argument(
        "--tabular-summary",
        default="./runs/lightweight_tabular_full_3seed/baselines_seed_summary_medium.csv",
    )
    ap.add_argument(
        "--tabular-metrics",
        default="./runs/lightweight_tabular_full_3seed/baselines_metrics_medium.csv",
    )
    ap.add_argument(
        "--outdir",
        default="./runs/lightweight_paper_report",
    )
    return ap.parse_args()


def aggregate_text_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["text_col"] == "text"].copy()
    grouped = (
        df.groupby(["method_name", "text_col", "val_experiment"], as_index=False)
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
    clean = (
        grouped[grouped["val_experiment"] == "E1_clean"][["method_name", "macro_f1_mean"]]
        .rename(columns={"macro_f1_mean": "macro_f1_base"})
        .copy()
    )
    grouped = grouped.merge(clean, on="method_name", how="left")
    grouped["delta_macro_f1_mean"] = grouped["macro_f1_mean"] - grouped["macro_f1_base"]
    grouped["delta_macro_f1_std"] = 0.0
    grouped["method_family"] = "text_baseline"
    grouped["source_run"] = Path(path).parent.name
    return grouped


def aggregate_tabular_summary(summary_path: Path, metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path).copy()
    if "delta_macro_f1_mean" not in df.columns:
        clean = (
            df[df["val_experiment"] == "E1_clean"][["method_name", "macro_f1_mean"]]
            .rename(columns={"macro_f1_mean": "macro_f1_base"})
            .copy()
        )
        df = df.merge(clean, on="method_name", how="left")
        df["delta_macro_f1_mean"] = df["macro_f1_mean"] - df["macro_f1_base"]
        df["delta_macro_f1_std"] = 0.0
    meta = pd.read_csv(metrics_path)
    meta = (
        meta.groupby(["method_name", "val_experiment"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            total_params=("total_params", "mean"),
            trainable_params=("trainable_params", "mean"),
        )
        .fillna(0.0)
    )
    df = df.merge(meta, on=["method_name", "val_experiment"], how="left")
    df["method_family"] = "tabular"
    df["text_col"] = "structured_features"
    df["source_run"] = summary_path.parent.name
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
            "total_params",
            "trainable_params",
            "source_run",
        ]
    ].copy()


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["val_rank"] = out["val_experiment"].map(VAL_ORDER).fillna(999).astype(int)
    out = out.sort_values(
        ["val_rank", "macro_f1_mean", "method_family", "method_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return out.drop(columns=["val_rank"])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text_df = aggregate_text_summary(Path(args.text_summary))
    tabular_df = aggregate_tabular_summary(Path(args.tabular_summary), Path(args.tabular_metrics))
    all_df = sort_results(pd.concat([text_df, tabular_df], ignore_index=True, sort=False))

    clean_df = (
        all_df[all_df["val_experiment"] == "E1_clean"]
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    recommended_df = all_df[
        all_df["method_name"].isin(["tfidf_svm", "MLP_MaskAug", "MLP"])
    ].copy()
    recommended_df = sort_results(recommended_df)

    all_df.to_csv(outdir / "lightweight_all_results__medium.csv", index=False)
    clean_df.to_csv(outdir / "lightweight_main_clean_table__medium.csv", index=False)
    all_df.to_csv(outdir / "lightweight_robustness_table__medium.csv", index=False)
    recommended_df.to_csv(outdir / "lightweight_recommended_methods__medium.csv", index=False)

    print(f"[REPORT] saved -> {outdir}")


if __name__ == "__main__":
    main()
