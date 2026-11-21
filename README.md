# CardioRisk AI — Modular cross-fitted PyTorch MLP

A Streamlit research application powered by one model: a PyTorch MLP (128–64–32). LightGBM is not included.

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

This is a modular single-process pipeline, not a set of separately deployed network services. Each stage can be tested and run independently.

## Leakage-safe evaluation

1. Validate the original semicolon-separated CSV at `data/cardio_train.csv`.
2. Apply deterministic row-level cleaning and feature engineering.
3. Split the dataset into stratified 80% development and 20% test partitions before cross-validation.
4. Run five-fold StratifiedKFold only on the development partition.
5. Fit imputation, one-hot encoding and scaling separately inside each fold.
6. Generate one out-of-fold prediction for each development row.
7. Select the high-recall threshold from OOF predictions subject to OOF precision >= 0.55.
8. Use inner validation in each fold for early stopping and use the median selected epoch count for final refitting.
9. Fit the final preprocessing configuration on all development rows and train a fresh MLP.
10. Evaluate the frozen pipeline once on the untouched test partition.

The test partition is never used for preprocessing, model selection, epoch selection or threshold selection.

## Artifacts

The GitHub Actions workflow saves these files to `artifacts/`:

- `mlp_state_dict.pt`
- `preprocessing.json`
- `metrics.json`
- `training_history.csv`\n- `research_validation.json` — cross-validated model comparison, confidence intervals, calibration diagnostics and subgroup analysis\n- `benchmark_oof_predictions.csv` — development OOF scores for benchmark models\n- `specificity_investigation.json` / `.csv` — class-weight and HistGradientBoosting comparisons at matched recall floors\n- `specificity_oof_predictions.csv` — development OOF probabilities for the specificity study\n- `benchmark_folds.csv` — fold-level benchmark results\n- `threshold_analysis.csv` — operating-point trade-offs

## Experimental MLP variants\n\n`mlp_variant_benchmark.py` compares compact, baseline and wider MLP configurations with different dropout and positive-class loss weights using three-fold development-only out-of-fold predictions. It is an experiment, not a deployed model selection step; candidates require independent evaluation before promotion.\n\n## Research dashboard

The Streamlit workspace includes held-out test metrics, fold-level validation, a recall-constrained operating-point explorer, calibration reliability bins, descriptive subgroup diagnostics, development-only benchmark comparisons, a specificity study, and a CatBoost/XGBoost study with sigmoid calibration. All new comparisons use five-fold development OOF predictions; no experimental candidate alters deployed inference. Matched-recall thresholds are exploratory because they are selected and summarized on pooled OOF scores.

## Deploy on Streamlit Community Cloud

The repository is prepared for Streamlit Community Cloud:

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **New app**.
3. Choose repository `Harshithpatali/cardiorisk-ai`, branch `main`, and main file path `app.py`.
4. Choose a supported Python version (Python 3.11 is a good starting point).
5. Select **Deploy**. The app loads the committed model and preprocessing artifacts from `artifacts/`; no secrets are required.

Do not remove the committed files in `artifacts/`. If the app fails to build, open **Manage app → Logs** and use the first dependency or import error to diagnose it.

## Run locally

```bash
pip install -r requirements.txt
python validate_data.py
python train.py
pytest -q
streamlit run app.py
```

Upload the original CSV to `data/cardio_train.csv`. See [data/README.md](data/README.md) for the required schema.


## Bayesian two-stage MLP cascade experiment

`bayesian_mlp_cascade.py` evaluates a research-only cascade without changing the deployed Streamlit inference path:

1. Stage 1 reuses the existing PyTorch MLP and sets its gate threshold for 99.5% development OOF recall, deliberately leaving recall headroom for Stage 2.
2. Stage 2 is a second PyTorch MLP trained only on Stage-1-positive development rows. Its unweighted loss and a separate threshold are intended to reduce false positives.
3. Stage 2 uses 30-pass Monte Carlo dropout to report a predictive mean and an approximate epistemic-uncertainty estimate.
4. A Bayesian Beta-Binomial decision layer starts with a Beta(1,1) prior for precision. It selects the Stage-2 threshold that maximizes the 95% posterior lower credible bound while satisfying the requested overall cascade recall target of 95%, 97%, or 98%.
5. The complete cascade is evaluated once on the untouched test partition. Results are written to `artifacts/mlp_bayesian_cascade.json`, `artifacts/mlp_bayesian_cascade_summary.csv`, and `artifacts/mlp_bayesian_cascade_folds.csv`.

The Bayesian interval is conditional on the sampled dataset and selected threshold; it is not a clinical confidence statement. Because Stage 2 sees the same tabular variables as Stage 1, it is a learned second-pass filter—not an independent confirmatory medical test. The cascade is an experiment and is not promoted into deployed inference automatically.

## Limitations

The public dataset is cross-sectional. There is no external or temporal validation, clinical validation, or independent population replication. Internal cross-validation and subgroup diagnostics do not establish fairness or generalisability. Calibration metrics are internal research diagnostics, not a clinical calibration study. A negative result must never be used to rule out disease or delay professional medical care.
