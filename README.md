# Heart Disease Prediction System

A binary classifier on the UCI Heart Disease (Cleveland) data, tuned for **recall first**: a missed sick
patient costs far more than a false alarm. Logistic regression, random forest and XGBoost are compared on
one split, the winner is served in a Streamlit form, and PostgreSQL records every training run.

> **Educational demo, not a diagnostic tool.** 303 patients, one clinic, 1980s data, 68% male.

## Results at a glance

| | Cleveland test (n=61) | Other UCI sites (n=617, frozen model) |
| --- | --- | --- |
| Recall | **0.96** (95% CI 0.88–1.00), 1 of 28 missed | **0.70** pooled; 0.74 Hungarian, 0.80 Swiss, 0.60 VA |
| Precision | 0.73 | 0.86 (prevalence differs, not comparable) |
| ROC-AUC | 0.96 | 0.83 pooled |

Three things worth taking from this:

1. **Logistic regression won.** Neither tree model beat it on CV recall by more than one standard deviation, so
   the simpler model ships. XGBoost's perfect test recall came from a threshold of 0.03 that flags half the
   healthy patients.
2. **The 61-row test set is noisy.** One missed patient moves recall by 3.6 points. The bootstrap interval
   spans 0.88 to 1.00, so the three models are effectively tied on recall.
3. **The recall target does not transfer.** On other sites recall drops to 0.70, and in cross-validation it is
   lower for women (0.78) than for men (0.93), with only 18 sick women in the training data. The external
   rows are missing `ca` (the model's second-strongest feature) 98% of the time, so this is a stress test,
   not a clean benchmark. See [reports/external_validation.md](reports/external_validation.md).

Full tables, odds ratios, permutation importance and plots: [reports/model_comparison.md](reports/model_comparison.md).

## Data traps found

- The popular Kaggle `heart.csv` has an **inverted target** (`target=1` means healthy) and hides missing values
  as fake codes (`ca=4`, `thal=0`). This repo loads the UCI file and a test fails if the label direction flips.
- The 1,025-row Kaggle copy is the same patients repeated, which leaks test rows into training. Published
  95%+ accuracy on this data is usually that leak. Any test accuracy above 0.92 triggers a leakage check here.

## Model card

- **Intended use:** learning and portfolio demonstration. Not for clinical decisions.
- **Model:** class-weighted logistic regression on one shared preprocessor (impute, scale, one-hot), threshold
  0.31 chosen as the highest cut-off with out-of-fold train recall of at least 0.90.
- **Data:** UCI Cleveland, 303 patients, 46% with disease. 6 missing values (`ca` 4, `thal` 2).
- **Known limits:** one clinic and era; weaker recall for women; probabilities come from a class-weighted model
  and are not calibrated, so the app shows a "risk score"; performance drops on other sites; a 61-patient test
  set cannot rank models reliably.

## Run it

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env
docker compose up -d          # postgres:16 on host port 5433
python -m src.ingest          # raw file -> patients_raw (idempotent; downloads the file if absent)
python -m src.train           # ~7 min: CV-tune 3 models, OOF thresholds, model_runs rows
python -m src.evaluate        # test metrics, plots, reports/model_comparison.md, activates the winner
python -m src.external        # frozen model on the other UCI sites -> reports/external_validation.md
streamlit run app/streamlit_app.py
pytest -q                     # DB tests skip when Postgres is not running
```

Host port 5433 avoids clashes with another local Postgres on 5432. Change it in `docker-compose.yml` and
`.env` together if you prefer. The app works without Postgres: it falls back to `models/metadata.json`.

## Layout

`src/` ingest, data, features, models, train, evaluate, external, inference · `app/streamlit_app.py` ·
`tests/` · `reports/` · `models/` (committed so the app runs on a fresh clone) · `notebooks/01_eda.ipynb`

## Data and licence

Data: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart-disease), CC BY 4.0. Code: MIT.
