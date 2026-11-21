# CardioRisk AI — Modular cross-fitted PyTorch MLP

A Streamlit research application powered in deployed inference by one model: a PyTorch MLP (128–64–32).

> Educational/research use only. This is not clinically validated and must not be used for diagnosis, treatment, screening, or care decisions. The dataset label is cross-sectional and is not a future disease-risk estimate.

## Modular project structure

Each pipeline responsibility has its own Python module. `train.py` orchestrates the baseline pipeline. The research cascade is isolated in `tree_stage2_cascade.py`.

## Leakage-safe evaluation

1. Validate and clean the original dataset.
2. Create a stratified development/test split before cross-validation.
3. Fit preprocessing inside each fold and generate development OOF MLP probabilities.
4. Select a high-recall Stage-1 gate from OOF predictions.
5. Fit the final Stage-1 MLP on development data and evaluate it on untouched test data.
6. Do not use in-sample Stage-1 predictions to train Stage 2.

## Two-stage selective verification experiment

Stage 1 is the existing PyTorch MLP and remains the deployed model. Its OOF development probabilities determine a gate selected for 99.5% development recall.

- Below the gate: Stage-1 negative; no Stage 2.
- Gate through below 0.50: borderline positive; route to Stage 2.
- At least 0.50: direct positive; bypass Stage 2.

The experiment saves `artifacts/stage1_prediction_dataset.csv`: every original feature plus actual label, Stage-1 prediction, positive-class confidence, data split, routing decision and Stage-2 training eligibility. Development rows use OOF predictions; test rows use predictions from the final refit MLP. **Only development OOF borderline rows are eligible for Stage-2 training.** Test rows are never used for training or threshold selection.

XGBoost and LightGBM are compared as Stage 2. Each is trained only on the borderline development population, using five outer folds and sigmoid calibration within each fold. Stage-2 thresholds are selected from development OOF predictions to satisfy end-to-end recall floors of 95%, 97%, 98%, 99%, and 99.5%. The 99% and 99.5% points specifically test whether Stage 2 can retain more of the high-recall MLP behavior while reducing false positives; all thresholds are selected on development OOF data and evaluated once on the untouched test set. A score is an estimated positive-class probability, not a guarantee that the individual is positive or that population precision equals that score. Calibration is assessed on the routed test subset.

The cascade's final positive rule is: Stage-1 confidence >= 0.50 OR (Stage-1 confidence in the borderline band AND Stage-2 score >= selected threshold). Cases below the Stage-1 gate cannot be recovered by Stage 2, so gate recall and total cascade false negatives are reported explicitly.

Artifacts:
- `tree_stage2_cascade_comparison.json/.csv`: XGBoost vs LightGBM and end-to-end test metrics.
- `tree_stage2_cascade_folds.csv`: outer fold counts.
- `stage1_prediction_dataset.csv`: complete Stage-1 prediction ledger with original features, labels, confidence, split and routing.
- `mlp_bayesian_cascade.*`, `mlp_matched_recall_audit.*`: earlier MLP-to-MLP research.

Neither candidate is promoted to Streamlit inference automatically. The deployed MLP and Render/Streamlit configuration remain unchanged. The dataset is cross-sectional and no result establishes clinical validity.

## Run locally

```bash
pip install -r requirements.txt
python validate_data.py
python train.py
python tree_stage2_cascade.py
pytest -q
streamlit run app.py
```

See [data/README.md](data/README.md) for the source schema.
