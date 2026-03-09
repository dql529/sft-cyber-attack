# Paper Revision Notes

## New Paper Direction

- Drop `LLM superiority` as the paper's main claim.
- Make the paper about `lightweight intrusion detection under missing and corrupted features`.
- Use `UNSW-NB15` as the primary benchmark and keep `UAV` only as deployment motivation unless a unified UAV evaluation pipeline is completed.

## Core Story

The defensible story is now:

1. Compact textified flow records are already handled very well by sparse lexical models such as `TF-IDF + SVM`.
2. A lightweight structured model can become competitive with that strong text baseline if it is trained with explicit missingness-aware augmentation.
3. The main gain is not "bigger model capacity", but `robustness to missing structured observations`.

This is a better SCI story than the old frozen-LLM narrative because it is aligned with the actual evidence and the available hardware.

## Primary Method

- Main proposed method: `MLP_MaskAug`
- Implementation: [nn_tabular_baseline.py](/d:/桌面/Model_Agent/nn_tabular_baseline.py)
- Method definition:
  - same structured input features as the baseline `MLP`
  - train-time random feature masking augmentation
  - train-time block missingness augmentation
  - no large-model fine-tuning
  - no expensive multimodal fusion

This should be presented as a `missingness-aware lightweight structured IDS`.

## Main Baselines

- Text baseline:
  - `TF-IDF + SVM`
  - `TF-IDF + Logistic Regression`
  - implementation: [text_tfidf_baseline.py](/d:/桌面/Model_Agent/text_tfidf_baseline.py)
- Structured baselines:
  - `MLP`
  - `Linear SVM`
  - `Logistic Regression`
  - implementation: [nn_tabular_baseline.py](/d:/桌面/Model_Agent/nn_tabular_baseline.py)
- Legacy LLM baselines:
  - frozen probes and fine-tuned BERT may be kept only as secondary context
  - do not mix old `runs/medium_main` LLM tables with the new lightweight tables unless those LLM baselines are rerun on the current prompt files

## Current Authoritative Result Folders

Use these as the paper's current source of truth:

- Text route: [summary__medium.csv](/d:/桌面/Model_Agent/runs/lightweight_text_medium_3seed/summary__medium.csv)
- Structured route: [baselines_seed_summary_medium.csv](/d:/桌面/Model_Agent/runs/lightweight_tabular_full_3seed/baselines_seed_summary_medium.csv)
- Unified paper tables: [lightweight_main_clean_table__medium.csv](/d:/桌面/Model_Agent/runs/lightweight_paper_report/lightweight_main_clean_table__medium.csv), [lightweight_robustness_table__medium.csv](/d:/桌面/Model_Agent/runs/lightweight_paper_report/lightweight_robustness_table__medium.csv)

## Key Quantitative Findings

From the current medium 3-seed runs:

- `MLP_MaskAug` clean `Macro-F1 = 0.7037`
- `TF-IDF + SVM` clean `Macro-F1 = 0.7010`
- `MLP` clean `Macro-F1 = 0.6766`

Under heavy missingness:

- `E3_mask_30`
  - `MLP_MaskAug = 0.6098`
  - `TF-IDF + SVM = 0.5364`
  - `MLP = 0.4675`
- `E3_block_30`
  - `MLP_MaskAug = 0.6266`
  - `TF-IDF + SVM = 0.5521`
  - `MLP = 0.5157`
- `E3_mask_50`
  - `MLP_MaskAug = 0.5599`
  - `TF-IDF + SVM = 0.4789`
  - `MLP = 0.3797`

Under noise:

- `E4_noise_30`
  - `TF-IDF + SVM = 0.6584`
  - `MLP_MaskAug = 0.6181`
  - `MLP = 0.5940`
- `E4_noise_50`
  - `TF-IDF + SVM = 0.6411`
  - `MLP_MaskAug = 0.5882`
  - `MLP = 0.5515`

Interpretation:

- `TF-IDF + SVM` remains a very strong clean and noise-tolerant baseline.
- `MLP_MaskAug` is the strongest method under missing-feature degradation.
- The gain over plain `MLP` is large enough to justify a method paper.

## CIC External Validation

The fast external validation on `CICIDS2017` supports the same main direction:

- clean:
  - `TF-IDF + SVM = 0.9855`
  - `MLP_MaskAug = 0.9766`
  - `MLP = 0.9697`
- missingness:
  - `E3_mask_30`: `MLP_MaskAug = 0.9326`, `MLP = 0.7212`, `TF-IDF + SVM = 0.6728`
  - `E3_block_30`: `MLP_MaskAug = 0.9335`, `MLP = 0.7235`, `TF-IDF + SVM = 0.6894`
  - `E3_mask_50`: `MLP_MaskAug = 0.8789`, `MLP = 0.6393`, `TF-IDF + SVM = 0.3627`
- noise:
  - `E4_noise_30`: `MLP_MaskAug = 0.9721`, `MLP = 0.9463`, `TF-IDF + SVM = 0.3374`

Interpretation:

- `CIC` is not blocking evidence; it is positive external validation.
- The same high-level conclusion remains true:
  `missingness-aware lightweight structured modeling is more robust than both plain MLP and the sparse text baseline under degraded observations.`

## Defensible Claims

These are safe to write:

1. `Simple text baselines are strong on compact textified IDS records and should not be omitted.`
2. `Missingness-aware augmentation substantially improves a lightweight structured MLP under both random and block feature loss.`
3. `The proposed lightweight structured method reaches text-baseline-level clean performance while outperforming it under severe feature missingness.`
4. `For resource-constrained settings, lightweight structured modeling is more practical than large-model adaptation.`

## Claims That Should Not Be Made

Do not write any of these:

1. `Frozen LLM is the best method overall.`
2. `This is a UAV-native benchmark evaluation.`
3. `The old LLM tables and the new lightweight tables are directly comparable without rerunning the old baselines.`
4. `The proposed method is best under all degradations.`  
   `TF-IDF + SVM` is still stronger under the current noise settings.

## Suggested Hypotheses

- `H1`: A lightweight structured MLP trained with missingness-aware augmentation can close the clean-performance gap to strong text baselines.
- `H2`: Missingness-aware augmentation yields much larger gains under feature loss than under additive corruption, indicating that observation loss is the dominant failure mode.
- `H3`: On compact flow representations, robustness depends more on degradation-aware training than on model scale.

## Recommended Experiments To Keep

For the main paper:

- `TF-IDF + SVM`
- `TF-IDF + Logistic Regression`
- `MLP`
- `MLP_MaskAug`
- optional: `Linear SVM`, `Logistic Regression` on structured inputs as secondary baselines

The main evaluation grid should be:

- `E1_clean`
- `E3_mask_10 / 30 / 50`
- `E3_block_10 / 30 / 50`
- `E4_noise_10 / 30 / 50`

## Recommended Paper Angle

Title direction:

- `Missingness-Aware Lightweight Intrusion Detection Under Structured Feature Loss`
- `When Simple Models Are Strong: Lightweight Robust IDS with Missingness-Aware Augmentation`
- `A Lightweight Missingness-Robust MLP for Network Intrusion Detection`

The right story is no longer "LLMs are strong". The right story is:

`A lightweight structured model can match strong text baselines on clean data and surpass them under severe missing-feature corruption when trained with explicit missingness-aware augmentation.`
