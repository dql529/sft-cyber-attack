from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_EXPERIMENTS = [
    "E1_clean",
    "E3_mask_30",
    "E3_mask_50",
    "E3_block_30",
    "E3_block_50",
    "E4_noise_30",
    "E4_noise_50",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation-summaries",
        default=",".join(
            [
                "./runs/lightweight_tabular_full_3seed/baselines_seed_summary_medium.csv",
                "./runs/lightweight_ablation_maskonly_3seed/baselines_seed_summary_medium.csv",
                "./runs/lightweight_ablation_blockonly_3seed/baselines_seed_summary_medium.csv",
                "./runs/lightweight_ablation_combo_3seed/baselines_seed_summary_medium.csv",
            ]
        ),
    )
    ap.add_argument(
        "--split-metrics",
        default=",".join(
            [
                "./runs/lightweight_tabular_split42_key/baselines_metrics_medium.csv",
                "./runs/lightweight_tabular_split52_key/baselines_metrics_medium.csv",
                "./runs/lightweight_tabular_split62_key/baselines_metrics_medium.csv",
            ]
        ),
    )
    ap.add_argument("--outdir", default="./runs/lightweight_fasttrack_report")
    return ap.parse_args()


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_ablation_table(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if path.parent.name == "lightweight_tabular_full_3seed":
            df = df[df["method_name"] == "MLP"].copy()
        df = df[df["method_name"].isin(["MLP", "MLP_MaskOnly", "MLP_BlockOnly", "MLP_MaskAug"])].copy()
        df = df[df["val_experiment"].isin(KEY_EXPERIMENTS)].copy()
        df["source_run"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    order = {name: i for i, name in enumerate(KEY_EXPERIMENTS)}
    out["val_rank"] = out["val_experiment"].map(order).fillna(999).astype(int)
    out = out.sort_values(["val_rank", "macro_f1_mean"], ascending=[True, False]).drop(columns=["val_rank"])
    return out.reset_index(drop=True)


def build_split_table(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["method_name"].isin(["MLP", "MLP_MaskAug"])].copy()
        df = df[df["val_experiment"].isin(KEY_EXPERIMENTS)].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    agg = (
        merged.groupby(["split_seed", "method_name", "val_experiment"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_model_seeds=("seed", "nunique"),
        )
        .fillna(0.0)
    )
    base = (
        agg[agg["val_experiment"] == "E1_clean"][["split_seed", "method_name", "macro_f1_mean"]]
        .rename(columns={"macro_f1_mean": "macro_f1_base"})
        .copy()
    )
    agg = agg.merge(base, on=["split_seed", "method_name"], how="left")
    agg["delta_macro_f1_mean"] = agg["macro_f1_mean"] - agg["macro_f1_base"]
    order = {name: i for i, name in enumerate(KEY_EXPERIMENTS)}
    agg["val_rank"] = agg["val_experiment"].map(order).fillna(999).astype(int)
    agg = agg.sort_values(["split_seed", "val_rank", "macro_f1_mean"], ascending=[True, True, False])
    return agg.drop(columns=["val_rank"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ablation = build_ablation_table(parse_list(args.ablation_summaries))
    split = build_split_table(parse_list(args.split_metrics))

    if not ablation.empty:
        ablation.to_csv(outdir / "ablation_table__medium.csv", index=False)
    if not split.empty:
        split.to_csv(outdir / "split_table__medium.csv", index=False)

    print(f"[FASTTRACK REPORT] saved -> {outdir}")


if __name__ == "__main__":
    main()
