# CardioRisk AI — Cross-fitted high-recall PyTorch MLP

A Streamlit research demonstration powered by **one model only: a PyTorch MLP (128–64–32)**. LightGBM is not part of this project.

> **Safety:** Educational/research software only. It is not clinically validated and must not be used for diagnosis, treatment, screening, or care decisions. The dataset label is cross-sectional; the output is not future disease risk or an individual's clinical probability.

## Leakage-safe training and evaluation

GitHub Actions reads the original semicolon-separated dataset from `data/cardio_train.csv`, validates it, and runs the MLP training pipeline.

1. **Holdout first:** stratified 80/20 train/test split before cross-validation. The test partition is isolated.
2. **Five-fold cross-fitting:** `StratifiedKFold(5)` runs only on the training partition. Each row in the training partition receives one out-of-fold (OOF) prediction from a model that did not train on that row.
3. **Fold-local preprocessing:** numeric median imputation and standardization, categorical imputation and one-hot encoding, and binary imputation are fitted separately inside each fold's training data. No scaler or category vocabulary is fitted on validation/test rows.
4. **Epoch selection:** each fold uses an inner stratified validation split for early stopping. The selected epoch counts are aggregated by their median.
5. **Threshold selection:** the high-recall threshold is chosen from OOF training predictions only, maximizing recall subject to OOF precision >= 0.55.
6. **Final refit:** preprocessing is fitted on the full training partition, then a fresh MLP is trained for the median selected epoch count.
7. **One final test evaluation:** the frozen model and OOF-selected threshold are evaluated on the untouched test partition once.

The test partition is not used for preprocessing, epoch selection, threshold selection, feature selection, or hyperparameter tuning.

## Published artifacts

The workflow commits these generated files to `artifacts/`:

- `mlp_state_dict.pt` — final MLP weights
- `preprocessing.json` — final training-fitted imputation, scaling and category configuration
- `metrics.json` — fold metrics, OOF metrics, threshold provenance and untouched test metrics
- `training_history.csv` — inner-validation training histories by fold

The workflow runs on pushes affecting training code and can also be started manually from **GitHub → Actions → Train and publish high-recall MLP → Run workflow**. It uses the repository's `GITHUB_TOKEN` with contents-write permission to save artifacts.

## Dataset

Upload the original CSV to `data/cardio_train.csv`. It must be semicolon-separated and contain the expected columns documented in [data/README.md](data/README.md).

## High-recall policy and limitations

The operating threshold prioritizes recall subject to an OOF precision constraint. High recall may produce many false positives. This is an experimental decision rule, not a clinically validated screening policy. Model scores are uncalibrated and must not be interpreted as disease probabilities.

One public cross-sectional dataset; no external/temporal validation, calibration study, confidence intervals, subgroup fairness audit, or clinical validation. A negative result must never be used to rule out disease or delay professional medical care.

## Run locally

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```
