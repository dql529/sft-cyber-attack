from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


FINAL_KEEP = {
    "unsw_main": [
        "runs/lightweight_paper_report/lightweight_main_clean_table__medium.csv",
        "runs/lightweight_paper_report/lightweight_robustness_table__medium.csv",
        "runs/lightweight_paper_report/lightweight_recommended_methods__medium.csv",
    ],
    "unsw_fasttrack": [
        "runs/lightweight_fasttrack_report/ablation_table__medium.csv",
        "runs/lightweight_fasttrack_report/ablation_degraded_summary__medium.csv",
        "runs/lightweight_fasttrack_report/split_table__medium.csv",
        "runs/lightweight_fasttrack_report/split_significance__medium.csv",
    ],
    "cic_external": [
        "runs/cic_external_report/cic_external_table__medium.csv",
        "runs/cic_external_report/cic_external_manifest.json",
        "runs/cic_external_text/summary__medium.csv",
        "runs/cic_external_tabular/baselines_seed_summary_medium.csv",
    ],
}

NOTES_KEEP = [
    "paper_revision_notes.md",
    "paper_hypotheses_current.md",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="./deliverables/sci_upload_bundle")
    ap.add_argument("--final-runs-root", default="./runs/final_results")
    ap.add_argument("--cleanup", action="store_true")
    return ap.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    bundle_dir = (root / args.bundle_dir).resolve()
    final_runs_root = (root / args.final_runs_root).resolve()
    tables_dir = bundle_dir / "tables"
    figures_dir = bundle_dir / "figures"
    notes_dir = bundle_dir / "notes"
    logs_dir = bundle_dir / "logs"
    for path in [tables_dir, figures_dir, notes_dir, logs_dir, final_runs_root]:
        path.mkdir(parents=True, exist_ok=True)

    retained = []
    for group, rel_paths in FINAL_KEEP.items():
        group_dir = final_runs_root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for rel in rel_paths:
            src = (root / rel).resolve()
            if not src.exists():
                continue
            copy_file(src, group_dir / src.name)
            copy_file(src, tables_dir / src.name)
            retained.append(str(src))

    for rel in NOTES_KEEP:
        src = (root / rel).resolve()
        if src.exists():
            copy_file(src, notes_dir / src.name)
            retained.append(str(src))

    manifest = {
        "final_runs_root": str(final_runs_root),
        "bundle_dir": str(bundle_dir),
        "retained_source_files": retained,
        "cleanup_enabled": bool(args.cleanup),
        "no_hidden_context_file_added": True,
    }
    with open(logs_dir / "result_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(logs_dir / "cleanup_log.txt", "w", encoding="utf-8") as f:
        f.write("Final result packaging completed.\n")
        f.write(f"cleanup_enabled={bool(args.cleanup)}\n")

    if args.cleanup:
        runs_root = root / "runs"
        keep_run_dirs = {"final_results"}
        for child in runs_root.iterdir():
            if child.name in keep_run_dirs:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        for pycache in root.rglob("__pycache__"):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)

        generated_data_dirs = [
            root / "data" / "cic_binary_structured",
        ]
        for path in generated_data_dirs:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    print(f"[PACKAGE] bundle -> {bundle_dir}")
    print(f"[PACKAGE] final runs -> {final_runs_root}")


if __name__ == "__main__":
    main()
