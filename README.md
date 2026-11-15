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
- `training_history.csv`

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

## Limitations

The public dataset is cross-sectional. There is no external or temporal validation, calibration study, confidence interval analysis, subgroup fairness audit or clinical validation. Model scores are uncalibrated. A negative result must never be used to rule out disease or delay professional medical care.
