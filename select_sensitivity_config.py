from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_CONDITIONS = ["E1_clean", "E3_mask_30", "E3_block_30", "E4_noise_30"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select the final augmentation configuration under a fixed reviewer rule.")
    ap.add_argument("--inputs", required=True, help="Comma-separated ablation/sensitivity CSV paths.")
    ap.add_argument("--output-table", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--clean-threshold", type=float, default=0.01)
    return ap.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(Path(path).resolve()) for path in parse_csv_list(args.inputs)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["val_experiment"].isin(CORE_CONDITIONS)].copy()
    summary = (
        df.groupby("method_name", as_index=False)
        .agg(
            clean_macro_f1=("macro_f1_mean", lambda s: float(s[df.loc[s.index, "val_experiment"] == "E1_clean"].iloc[0])),
            core_mean_macro_f1=("macro_f1_mean", "mean"),
        )
        .sort_values(["clean_macro_f1", "core_mean_macro_f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    best_clean = float(summary["clean_macro_f1"].max())
    eligible = summary[summary["clean_macro_f1"] >= best_clean - args.clean_threshold].copy()
    eligible = eligible.sort_values(["core_mean_macro_f1", "clean_macro_f1"], ascending=[False, False]).reset_index(drop=True)
    selected = eligible.iloc[0]

    out_table = Path(args.output_table).resolve()
    out_table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_table, index=False)

    payload = {
        "selection_rule": {
            "core_conditions": CORE_CONDITIONS,
            "clean_threshold": args.clean_threshold,
        },
        "best_clean_macro_f1": best_clean,
        "selected_method_name": str(selected["method_name"]),
        "selected_clean_macro_f1": float(selected["clean_macro_f1"]),
        "selected_core_mean_macro_f1": float(selected["core_mean_macro_f1"]),
        "eligible_methods": eligible["method_name"].tolist(),
    }
    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SELECT] saved -> {out_json}")


if __name__ == "__main__":
    main()
