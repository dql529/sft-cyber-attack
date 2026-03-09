# Current SCI-Safe Hypotheses

This file records hypotheses that are defensible given the current datasets and results.

## Supported by current UNSW results

1. `Same-input sparse lexical baselines can outperform frozen LLM probes on compact textified flow records.`
   - Current evidence: `TF-IDF + SVM` beats all frozen probes on clean UNSW.
   - This is the strongest supported statement for the current text pipeline.

2. `Frozen LLM probes remain competitive with fine-tuned BERT and outperform the current structured baselines on the filtered UNSW feature set.`
   - Current evidence: `deepseek13` is above the current tabular models, and close to fine-tuned BERT in macro-F1.
   - This is weaker than "best overall", but still publishable if framed correctly.

3. `Textified IDS pipelines are substantially more fragile to missing features than to moderate numeric noise.`
   - Current evidence: large drops under `E3_mask_*`, much smaller drops under `E4_noise_*`.

## Plausible but not yet tested

4. `The current prompt design suppresses the semantic advantage of pretrained text encoders.`
   - Rationale: the prompt is a short binned `key=value` list, which is close to a sparse lexical classification problem.
   - Needed test: compare raw-value text, binned text, and natural-language text against the same baselines.

5. `Pretrained text encoders may show value mainly in cross-dataset transfer, not in same-dataset in-domain testing.`
   - Rationale: sparse lexical models often dominate in-domain, but transfer can favor pretrained representations.
   - Needed test: binary cross-dataset experiments across UNSW, CICIDS2017, and UAV.

6. `Feature filtering can change the ranking between LLM and traditional baselines.`
   - Rationale: the current curated UNSW dataset has only 16 columns, and the prompt keeps an even smaller semantic surface after binning.
   - Needed test: compare `Augmented-UNSW_NB15` vs `unsw15_filtered_nolog`.

## Not supported and should not be claimed

7. `Frozen LLM is the best method overall.`
   - False under the current same-input comparison.

8. `The current method is robust to missing features.`
   - False under the current masking results.

9. `This is already a validated UAV IDS benchmark study.`
   - False today, because only the UAV raw dataset is present; no unified evaluation pipeline has been completed on it yet.
