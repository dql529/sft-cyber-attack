from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABEL_MAP


ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "deliverables" / "sci_upload_bundle"
TABLES_MAIN_DIR = BUNDLE_DIR / "tables_main"
TABLES_APPENDIX_DIR = BUNDLE_DIR / "tables_appendix"
FIGURES_MAIN_DIR = BUNDLE_DIR / "figures_main"
FIGURES_APPENDIX_DIR = BUNDLE_DIR / "figures_appendix"
PER_CLASS_DIR = BUNDLE_DIR / "per_class"
LOGS_DIR = BUNDLE_DIR / "logs"

CONDITION_ORDER = ["Clean", "Mask30", "Mask50", "Block30", "Block50", "Noise30", "Noise50"]
NON_CLEAN_ORDER = ["Mask30", "Mask50", "Block30", "Block50", "Noise30", "Noise50"]
METHOD_ORDER = ["MLP MaskAug", "MLP", "TF-IDF + Logistic Regression"]
METHOD_LABELS = {
    "MLP MaskAug": "MLP MaskAug",
    "MLP": "MLP",
    "TF-IDF + Logistic Regression": "TF-IDF + LR",
}
COLORS = {
    "MLP MaskAug": "#0B5FA5",
    "MLP": "#8A8F98",
    "TF-IDF + Logistic Regression": "#2E8B57",
}
MARKERS = {
    "MLP MaskAug": "o",
    "MLP": "s",
    "TF-IDF + Logistic Regression": "D",
}
LINESTYLES = {
    "MLP MaskAug": "-",
    "MLP": "--",
    "TF-IDF + Logistic Regression": "-.",
}
UNSW_CLASS_ORDER = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
GROUPS = [
    ("Random masking", ["Clean", "Mask30", "Mask50"]),
    ("Block masking", ["Clean", "Block30", "Block50"]),
    ("Noise", ["Clean", "Noise30", "Noise50"]),
]


def save_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def load_table(name: str, appendix: bool = False) -> pd.DataFrame:
    base = TABLES_APPENDIX_DIR if appendix else TABLES_MAIN_DIR
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def robustness_frame(name: str) -> pd.DataFrame:
    df = load_table(name)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df.sort_values(["method_name", "condition"]).reset_index(drop=True)


def metric(df: pd.DataFrame, method: str, condition: str) -> float:
    row = df[(df["method_name"] == method) & (df["condition"] == condition)]
    if row.empty:
        raise KeyError(f"Missing {method} / {condition}")
    return float(row.iloc[0]["macro_f1_mean"])


def build_missingness_mean(df: pd.DataFrame, method: str) -> float:
    cols = ["Mask30", "Mask50", "Block30", "Block50"]
    return float(np.mean([metric(df, method, c) for c in cols]))


def style_common_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_faceted_robustness(df: pd.DataFrame, dataset_label: str, stem: str) -> str:
    y_vals = [float(x) for x in df["macro_f1_mean"].tolist()]
    y_min = max(0.0, min(y_vals) - 0.04)
    y_max = min(1.0, max(y_vals) + 0.03)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True, constrained_layout=True)
    legend_handles: list | None = None

    for ax, (group_title, conditions) in zip(axes, GROUPS):
        x = np.arange(len(conditions))
        for method in METHOD_ORDER:
            scores = [metric(df, method, cond) for cond in conditions]
            lines = ax.plot(
                x,
                scores,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.2,
                markersize=6.3,
                label=METHOD_LABELS[method],
            )
            if legend_handles is None:
                legend_handles = lines
        ax.set_title(group_title, fontsize=10)
        ax.set_xticks(x, ["Clean", "30%", "50%"])
        ax.set_ylim(y_min, y_max)
        style_common_axis(ax)

    axes[0].set_ylabel("Macro-F1")
    fig.suptitle(dataset_label, y=1.05, fontsize=12, fontweight="bold")
    fig.legend(
        legend_handles if legend_handles else [],
        [METHOD_LABELS[m] for m in METHOD_ORDER],
        ncols=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )
    return save_pdf(fig, FIGURES_MAIN_DIR, stem)


def plot_gain_effect(unsw_df: pd.DataFrame, cic_df: pd.DataFrame, split_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=True, constrained_layout=True)

    split_lookup = {str(row["condition"]): row for _, row in split_df.iterrows()}

    for ax, dataset_name, df in zip(axes, ["UNSW-NB15", "CICIDS2017"], [unsw_df, cic_df]):
        y = np.arange(len(NON_CLEAN_ORDER))[::-1]
        ax.axvline(0.0, color="#444444", linewidth=1.0)
        ax.axhspan(3.5, 5.5, color="#F7FBFF", zorder=0)
        ax.axhspan(1.5, 3.5, color="#FAFAFA", zorder=0)
        ax.axhspan(-0.5, 1.5, color="#F5FBF7", zorder=0)

        for offset, method in [(-0.12, "MLP MaskAug"), (0.12, "TF-IDF + Logistic Regression")]:
            if dataset_name == "UNSW-NB15" and method == "MLP MaskAug":
                gains = np.array([float(split_lookup[cond]["mean_gain"]) for cond in NON_CLEAN_ORDER])[::-1]
            else:
                gains = np.array([metric(df, method, cond) - metric(df, "MLP", cond) for cond in NON_CLEAN_ORDER])[::-1]
            ypos = y + offset
            ax.plot(
                gains,
                ypos,
                marker=MARKERS[method],
                linestyle="None",
                markersize=7.2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
            ax.hlines(ypos, 0.0, gains, color=COLORS[method], linewidth=1.3, alpha=0.8, zorder=2)
            if dataset_name == "UNSW-NB15" and method == "MLP MaskAug":
                for gain, y_pos, cond in zip(gains, ypos, NON_CLEAN_ORDER[::-1]):
                    row = split_lookup[cond]
                    low = float(row["ci95_low"])
                    high = float(row["ci95_high"])
                    ax.errorbar(
                        gain,
                        y_pos,
                        xerr=np.array([[gain - low], [high - gain]]),
                        fmt="none",
                        ecolor=COLORS[method],
                        elinewidth=1.2,
                        capsize=2.8,
                        zorder=4,
                    )

        ax.set_title(dataset_name, fontsize=11)
        ax.set_yticks(y, NON_CLEAN_ORDER[::-1])
        style_common_axis(ax)

    axes[0].set_ylabel("Condition")
    axes[0].set_xlabel(r"$\Delta$ Macro-F1 over plain MLP")
    axes[1].set_xlabel(r"$\Delta$ Macro-F1 over plain MLP")
    axes[1].legend(frameon=False, loc="lower right")
    return save_pdf(fig, FIGURES_MAIN_DIR, "gain_over_mlp")


def plot_efficiency_tradeoff(unsw_df: pd.DataFrame, cic_df: pd.DataFrame, eff_df: pd.DataFrame) -> str:
    eff_df = eff_df.copy()
    eff_df["paper_method"] = eff_df["paper_method"].astype(str)
    robust_unsw = {m: build_missingness_mean(unsw_df, m) for m in METHOD_ORDER}
    robust_cic = {m: build_missingness_mean(cic_df, m) for m in METHOD_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=True, constrained_layout=True)

    for ax, dataset_name, robust_map in zip(axes, ["UNSW-NB15", "CICIDS2017"], [robust_unsw, robust_cic]):
        for method in METHOD_ORDER:
            row = eff_df[eff_df["paper_method"] == method]
            if row.empty:
                continue
            latency = float(row.iloc[0]["cpu_inference_latency_ms"])
            size_mb = float(row.iloc[0]["model_size_mb"])
            train_time = float(row.iloc[0]["training_time_s"])
            score = robust_map[method]
            bubble = 1400 * np.sqrt(size_mb + 0.03)
            ax.scatter(
                latency,
                score,
                s=bubble,
                color=COLORS[method],
                alpha=0.82,
                edgecolor="white",
                linewidth=1.0,
                marker=MARKERS[method],
            )
            ax.annotate(
                f"{METHOD_LABELS[method]}\n{train_time:.1f}s train",
                (latency, score),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8.5,
            )
        ax.set_xscale("log")
        ax.set_title(dataset_name, fontsize=11)
        ax.set_xlabel("CPU latency per sample (ms, log scale)")
        style_common_axis(ax)

    axes[0].set_ylabel("Mean Macro-F1 on missingness conditions")
    fig.suptitle("Efficiency-robustness trade-off", y=1.04, fontsize=12, fontweight="bold")
    return save_pdf(fig, FIGURES_MAIN_DIR, "efficiency_tradeoff")


def plot_relative_drop_heatmap(unsw_df: pd.DataFrame, cic_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    vmax = 0.0
    heatmaps = []
    for df in [unsw_df, cic_df]:
        matrix = []
        for method in METHOD_ORDER:
            clean = metric(df, method, "Clean")
            matrix.append([clean - metric(df, method, cond) for cond in NON_CLEAN_ORDER])
        arr = np.array(matrix)
        vmax = max(vmax, float(arr.max()))
        heatmaps.append(arr)

    for ax, dataset_name, arr in zip(axes, ["UNSW-NB15", "CICIDS2017"], heatmaps):
        im = ax.imshow(arr, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(NON_CLEAN_ORDER)), NON_CLEAN_ORDER, rotation=20)
        ax.set_yticks(range(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER])
        ax.set_title(dataset_name, fontsize=11)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                color = "white" if arr[i, j] > 0.5 * vmax else "#1f1f1f"
                ax.text(j, i, f"{arr[i, j]:.03f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(r"$\Delta_{\mathrm{drop}} = F1_{\mathrm{clean}} - F1_{\mathrm{degraded}}$")
    return save_pdf(fig, FIGURES_APPENDIX_DIR, "relative_drop_from_clean")


def plot_sorted_gain_bars(csv_name: str, value_col: str, stem: str, title: str, x_label: str) -> str:
    df = load_table(csv_name, appendix=True).copy()
    df = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    values = df[value_col].astype(float)
    colors = ["#B03A2E" if v < 0 else "#0B5FA5" for v in values]

    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    ax.barh(df["class"], values, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    for idx, val in enumerate(values):
        offset = 0.004 if val >= 0 else -0.004
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, idx, f"{val:.03f}", va="center", ha=ha, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    style_common_axis(ax)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)
    return save_pdf(fig, FIGURES_APPENDIX_DIR, stem)


def plot_confusion_matrix_mask30() -> str | None:
    cm_path = PER_CLASS_DIR / "confusion_matrix__UNSW__MLP_MaskAug__Mask30.npy"
    if not cm_path.exists():
        return None
    cm = np.load(cm_path).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(UNSW_CLASS_ORDER)), UNSW_CLASS_ORDER, rotation=40, ha="right")
    ax.set_yticks(range(len(UNSW_CLASS_ORDER)), UNSW_CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("UNSW confusion matrix for MLP MaskAug at Mask30", fontsize=11)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            text = f"{100 * norm[i, j]:.0f}%" if norm[i, j] >= 0.05 else ""
            color = "white" if norm[i, j] > 0.55 else "#1f1f1f"
            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")
    return save_pdf(fig, FIGURES_APPENDIX_DIR, "confusion_matrix_unsw_mask30")


def update_manifest(main_paths: list[str], appendix_paths: list[str]) -> None:
    manifest_path = LOGS_DIR / "result_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    generated_assets = manifest.get("generated_assets", [])
    refreshed = [
        str(BUNDLE_DIR / "notes" / "lightweight_paper_sections.tex"),
        *main_paths,
        *appendix_paths,
    ]
    manifest["generated_assets"] = list(dict.fromkeys(refreshed + generated_assets))
    manifest["main_text_figure_files"] = main_paths
    manifest["appendix_figure_files"] = appendix_paths
    manifest["paper_main_figures"] = len(main_paths)
    manifest["paper_appendix_figures"] = len(appendix_paths)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    unsw_df = robustness_frame("lightweight_robustness_table__medium.csv")
    cic_df = robustness_frame("cic_external_table__medium.csv")
    split_df = load_table("split_significance__medium.csv")
    eff_df = load_table("efficiency_table__medium.csv")

    main_paths = [
        plot_faceted_robustness(unsw_df, "UNSW-NB15: robustness by degradation family", "unsw_robustness_curve"),
        plot_gain_effect(unsw_df, cic_df, split_df),
        plot_faceted_robustness(cic_df, "CICIDS2017: robustness by degradation family", "cic_robustness_curve"),
        plot_efficiency_tradeoff(unsw_df, cic_df, eff_df),
    ]
    appendix_paths = [
        plot_relative_drop_heatmap(unsw_df, cic_df),
        plot_sorted_gain_bars(
            "per_class_f1_gain_mask30__medium.csv",
            "f1_gain_over_mlp",
            "per_class_f1_gain_mask30",
            "UNSW per-class F1 gain at Mask30",
            "F1 gain over MLP",
        ),
        plot_sorted_gain_bars(
            "per_class_recall_gain_block30__medium.csv",
            "recall_gain_over_mlp",
            "per_class_recall_gain_block30",
            "UNSW per-class recall gain at Block30",
            "Recall gain over MLP",
        ),
    ]
    cm_path = plot_confusion_matrix_mask30()
    if cm_path:
        appendix_paths.append(cm_path)

    update_manifest(main_paths, appendix_paths)
    print("[PLOT] journal-style figures regenerated")


if __name__ == "__main__":
    main()
