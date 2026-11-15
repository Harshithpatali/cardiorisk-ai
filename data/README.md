# Original dataset

Upload the original cardiovascular dataset into this folder as:

`data/cardio_train.csv`

Expected format: semicolon-separated CSV with these columns:

`id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio`

The training workflow reads this file when it is present. Do not upload personal or patient-identifiable medical records. The public benchmark dataset is intended for research/education only.

After uploading, run **Actions → Train and publish high-recall MLP → Run workflow**. The workflow validates the dataset, trains the MLP, evaluates it, runs tests, and commits model artifacts under `artifacts/`.
