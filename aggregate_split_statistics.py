from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata, wilcoxon
except Exception:  # pragma: no cover
    rankdata = None
    wilcoxon = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate paired split-level statistics for reviewer-ready reporting.")
    ap.add_argument("--inputs", required=True, help="Comma-separated baselines_metrics CSV paths.")
    ap.add_argument("--baseline-method", default="MLP")
    ap.add_argument("--main-method", default="MLP_MaskAug")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument(
        "--val-experiments",
        default="E1_clean,E3_mask_30,E3_mask_50,E3_block_30,E3_block_50,E4_noise_30,E4_noise_50",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def bootstrap_ci(values: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(float(np.mean(sample)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rank_biserial_from_diffs(diffs: np.ndarray) -> tuple[float, float, float]:
    nonzero = diffs[np.abs(diffs) > 0]
    if len(nonzero) == 0:
        return 0.0, 0.0, 0.0
    if rankdata is None:
        return float(np.sign(nonzero).mean()), np.nan, np.nan
    ranks = rankdata(np.abs(nonzero), method="average")
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = float(len(nonzero) * (len(nonzero) + 1) / 2)
    rbc = (w_pos - w_neg) / total if total > 0 else 0.0
    return float(rbc), w_pos, w_neg


def exact_wilcoxon_greater(diffs: np.ndarray) -> float:
    nonzero = diffs[np.abs(diffs) > 0]
    if len(nonzero) == 0:
        return 1.0
    if wilcoxon is None:
        if np.all(nonzero > 0):
            return float(1.0 / (2 ** len(nonzero)))
        return np.nan
    result = wilcoxon(nonzero, alternative="greater", zero_method="wilcox", method="exact")
    return float(result.pvalue)


def main() -> None:
    args = parse_args()
    csv_paths = [Path(x).resolve() for x in parse_csv_list(args.inputs)]
    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    keep_exps = parse_csv_list(args.val_experiments)
    df = df[df["val_experiment"].isin(keep_exps)].copy()

    base = df[df["method_name"] == args.baseline_method].copy()
    main_df = df[df["method_name"] == args.main_method].copy()
    key_cols = ["split_seed", "seed", "val_experiment"]
    merged = base[key_cols + ["macro_f1"]].merge(
        main_df[key_cols + ["macro_f1"]],
        on=key_cols,
        how="inner",
        suffixes=("_baseline", "_main"),
    )
    if merged.empty:
        raise RuntimeError("No paired rows found for split statistics.")

    rows = []
    for exp in keep_exps:
        sub = merged[merged["val_experiment"] == exp].copy()
        diffs = (sub["macro_f1_main"] - sub["macro_f1_baseline"]).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(diffs, args.bootstrap_iters, args.seed + len(rows))
        rbc, w_pos, w_neg = rank_biserial_from_diffs(diffs)
        rows.append(
            {
                "val_experiment": exp,
                "n_pairs": int(len(sub)),
                "mlp_mean_macro_f1": float(sub["macro_f1_baseline"].mean()),
                "maskaug_mean_macro_f1": float(sub["macro_f1_main"].mean()),
                "mean_gain": float(diffs.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "rank_biserial": rbc,
                "w_plus": w_pos,
                "w_minus": w_neg,
                "all_gains_positive": bool(np.all(diffs > 0)),
                "p_value_one_sided": exact_wilcoxon_greater(diffs),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[STATS] saved -> {out_path}")


if __name__ == "__main__":
    main()
