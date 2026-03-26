# Reproduction Layout

This repository is packaged as a lean reproducibility bundle for the final paper.
It intentionally keeps the core code, the minimal processed datasets, and the final paper assets,
while excluding large raw datasets, checkpoints, caches, and intermediate experiment runs.

## What Is Kept

- Core training and evaluation code:
  - `nn_tabular_baseline.py`
  - `text_tfidf_baseline.py`
  - `robustness_utils.py`
  - `measure_efficiency.py`
  - `plot_results.py`
  - `aggregate_split_statistics.py`
  - `select_sensitivity_config.py`
  - `build_reviewer_shield_assets.py`
  - `build_final_paper_assets.py`
- Minimal structured data required for the final paper:
  - `data/datasets/unsw_nb15/curated/unsw15_filtered_nolog.csv`
  - `data/prompt_csv/splits_unsw15_seed42.npz`
  - `data/prompt_csv/splits_unsw15_seed52.npz`
  - `data/prompt_csv/splits_unsw15_seed62.npz`
  - `data/cic_binary_structured_review/*.csv`
  - `data/cic_binary_structured_review/manifest.json`
- Final paper results and manuscript assets:
  - `deliverables/sci_upload_bundle/figures_main/`
  - `deliverables/sci_upload_bundle/figures_appendix/`
  - `deliverables/sci_upload_bundle/tables_main/`
  - `deliverables/sci_upload_bundle/tables_appendix/`
  - `deliverables/sci_upload_bundle/notes/lightweight_paper_sections.tex`

## What Is Intentionally Omitted

- Raw dataset archives and large raw CSV files.
- Cached prompt/text dumps from earlier experiments.
- `runs/reviewer_shield/` and other intermediate run directories.
- Model checkpoints (`.pt`) and intermediate metrics JSON files.
- Scratch files such as `tmp_pdf_extract.txt` and `__pycache__/`.

## Minimal Reproduction Steps

### 1. Rebuild the fairness-aligned UNSW text bundle

```powershell
python 01_data_preparation/build_unsw_fair_text_bundle.py
```

### 2. Rebuild the fairness-aligned CIC text bundle

```powershell
python 01_data_preparation/build_cic_fair_text_bundle.py
```

### 3. Run the main tabular baseline on UNSW

```powershell
python nn_tabular_baseline.py --model-type mlp_maskaugnoise --tier medium --seed 11
```

### 4. Run the aligned lexical baseline on UNSW

```powershell
python text_tfidf_baseline.py --prompt-dir data/fair_text_csv/seed42 --data-prefix unsw_structured_text --tier medium --dataset-name UNSW
```

### 5. Final paper assets

The upload-ready paper assets are already versioned in:

- `deliverables/sci_upload_bundle/figures_main/`
- `deliverables/sci_upload_bundle/figures_appendix/`
- `deliverables/sci_upload_bundle/tables_main/`
- `deliverables/sci_upload_bundle/tables_appendix/`

These are the authoritative outputs for the final manuscript. They are kept directly in the repository so that a fresh clone does not need to replay all historical intermediate runs.

### 6. About the omitted historical runs

```powershell
# Intentionally omitted from the lean repo:
# - runs/reviewer_shield/
# - historical checkpoints and metrics JSON files
```

## Notes

- The final paper uses the fairness-aligned comparison in which the strongest lexical baseline is
  `TF-IDF + Logistic Regression`.
- The final selected structured method is `MLP MaskAug`, implemented as the
  `mlp_maskaugnoise` training configuration.
- The authoritative upload-ready assets live under `deliverables/sci_upload_bundle/`.
- Legacy asset packers such as `build_reviewer_shield_assets.py` are kept for reference, but the
  lean remote bundle does not include all of their original intermediate inputs.
