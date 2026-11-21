# CardioRisk AI — Modular cross-fitted PyTorch MLP

A Streamlit research application powered in deployed inference by one model: a PyTorch MLP (128–64–32).

> Educational/research use only. This is not clinically validated and must not be used for diagnosis, treatment, screening, or care decisions. The dataset label is cross-sectional and is not a future disease-risk estimate.

## Modular project structure

Each pipeline responsibility has its own Python module. `train.py` only orchestrates the stages.

| Module | Responsibility |
|---|---|
| `config.py` | Shared paths and reproducible settings |
| `validate_data.py` | Validate source file, schema, target and row count |
| `preprocessing.py` | Deterministic feature engineering and fold-fitted transforms |
| `data_split.py` | Create the untouched stratified test split |
| `cross_validation.py` | Generate five-fold out-of-fold predictions |
| `model_training.py` | Train, early-stop, refit and predict with the MLP |
| `threshold_selection.py` | Select threshold from OOF predictions |
| `evaluation.py` | Calculate classification metrics |
| `save_artifacts.py` | Save weights, preprocessing, metrics and training history |
| `train.py` | Run the complete pipeline in order |
| `app.py` | Streamlit inference interface |

This is a modular single-process pipeline, not a set of separately deployed network services.

## Leakage-safe evaluation

1. Validate the original semicolon-separated CSV at `data/cardio_train.csv`.
2. Apply deterministic row-level cleaning and feature engineering.
3. Split into stratified 80% development and 20% test partitions before cross-validation.
4. Run five-fold StratifiedKFold only on the development partition.
5. Fit imputation, one-hot encoding and scaling separately inside each fold.
6. Generate one out-of-fold prediction for each development row.
7. Select the high-recall threshold from OOF predictions subject to OOF precision >= 0.55.
8. Use inner validation in each fold for early stopping and the median selected epoch count for final refitting.
9. Fit final preprocessing on all development rows and train a fresh MLP.
10. Evaluate the frozen pipeline once on the untouched test partition.

## Experiments and artifacts

GitHub Actions runs the baseline MLP pipeline, statistical audits, specificity and calibration studies, and the experimental cascade comparisons. Generated outputs are committed under `artifacts/`.

- `research_validation.json`, `benchmark_oof_predictions.csv`, `benchmark_folds.csv`: cross-validated model comparison and diagnostics.
- `specificity_investigation.*`: class-weight and tree comparisons at matched recall.
- `boosting_calibration_study.*`: CatBoost/XGBoost calibration study.
- `mlp_variant_benchmark.*`, `mlp_recall_study.*`, `mlp_operating_point_frontier.*`: MLP architecture, optimizer and threshold experiments.
- `mlp_bayesian_cascade.*`, `mlp_matched_recall_audit.*`: original MLP-to-MLP cascade and paired matched-recall audit.
- `tree_stage2_cascade_comparison.json/.csv`, `tree_stage2_cascade_folds.csv`: XGBoost versus LightGBM as the second-stage filter behind the unchanged MLP gate.

## XGBoost vs LightGBM second-stage experiment

`tree_stage2_cascade.py` compares both gradient-boosted tree classifiers as Stage 2. Stage 1 remains the existing PyTorch MLP and its gate is selected on development OOF predictions for 99.5% recall. Each tree model is trained on Stage-1-positive development rows, produces five-fold cross-fitted Stage-2 scores, and gets a Bayesian Beta-Binomial threshold selected to satisfy 95%, 97%, or 98% end-to-end OOF recall. The corresponding single-stage MLP threshold is calibrated to the same achieved OOF recall. Both systems are then compared on the same untouched test rows with 2,000 paired bootstrap resamples.

This experiment tests whether tree models are better second-pass filters than a second MLP. The posterior and bootstrap intervals are conditional on the selected thresholds and fitted models; they do not include model-training or threshold-selection uncertainty. Results are exploratory and do not establish clinical validity. **Neither tree cascade is automatically promoted into Streamlit inference.**

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **New app**.
3. Choose repository `Harshithpatali/cardiorisk-ai`, branch `main`, and main file path `app.py`.
4. Choose Python 3.11 and deploy. The app loads the committed MLP model and preprocessing artifacts from `artifacts/`.

The deployed inference path remains the single-stage MLP. Do not remove its committed artifacts.

## Run locally

```bash
pip install -r requirements.txt
python validate_data.py
python train.py
python tree_stage2_cascade.py
pytest -q
streamlit run app.py
```

Upload the original CSV to `data/cardio_train.csv`. See [data/README.md](data/README.md) for the required schema.

## Limitations

The public dataset is cross-sectional. There is no external or temporal validation, clinical validation, or independent population replication. Internal cross-validation and subgroup diagnostics do not establish fairness or generalisability. A negative result must never be used to rule out disease or delay professional medical care.
