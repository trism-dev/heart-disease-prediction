# CLAUDE.md — Heart Disease Prediction System

Binary classifier on the UCI Heart Disease (Cleveland) data. It compares logistic regression, random forest and XGBoost, is tuned for **recall first**, and is served in a Streamlit form. PostgreSQL (via SQLAlchemy) is the source of truth for patient data and training-run history. The full spec is in the PRD. This file is the working rulebook.

## Stack

Python 3.11+ · pandas · XGBoost ≥ 2.0 · scikit-learn ≥ 1.4 (LR, RF, pipelines, CV, metrics) · PostgreSQL 16 · SQLAlchemy 2.x + psycopg 3 · Streamlit · joblib, matplotlib, python-dotenv, pytest, ruff

## Prime directive

Missing a sick patient is far worse than a false alarm. Every modeling decision optimizes **recall** at a precision floor (≥ 0.70). Never select or report a model on accuracy alone, and never use a 0.5 threshold by default.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env         # DATABASE_URL=postgresql+psycopg://heart:heart@localhost:5432/heart
docker compose up -d         # local postgres:16
python -m src.ingest         # UCI file -> patients_raw (idempotent)
python -m src.train          # CV-tune 3 models, choose threshold, save models/
python -m src.evaluate       # test-set metrics, plots, reports/model_comparison.md
streamlit run app/streamlit_app.py
pytest -q -m "not db"        # fast tests, no database
pytest -q                    # full suite incl. @pytest.mark.db; must pass before any commit
ruff check . && ruff format .
```

## Repo map

```
src/config.py      paths, SEED=42, column lists, RECALL_TARGET=0.90, PRECISION_FLOOR=0.70, env loading
src/db.py          SQLAlchemy engine, session factory, ORM models: PatientRaw, ModelRun, Prediction
src/ingest.py      UCI file -> patients_raw ('?' -> NULL); idempotent
src/data.py        read patients_raw via SQLAlchemy -> validate -> binarise -> dedupe -> stratified split
src/features.py    build_preprocessor() -> ColumnTransformer
src/models.py      model factories + RandomizedSearchCV spaces
src/train.py       CV tuning, OOF threshold selection, save joblib, insert model_runs rows
src/evaluate.py    test metrics, bootstrap CIs, plots, report; update model_runs, set is_active
app/streamlit_app.py   loads the active model_runs row + its joblib; falls back to models/metadata.json
notebooks/01_eda.ipynb exploration only; no logic that src/ depends on
```

## Data: read this before touching anything

- **Source:** the UCI Heart Disease dataset, Cleveland subset (`processed.cleveland.data`, 303 rows), fetched via `ucimlrepo` (`fetch_ucirepo(id=45)`) or the raw file with `na_values="?"`. Save the untouched copy to `data/raw/` (git-ignored), then `src/ingest.py` loads it into `patients_raw`. **After ingest, all code reads from Postgres, never from the CSV.**
- **Do NOT trust Kaggle `heart.csv` variants.** The popular 303-row copy has an **inverted target** (`target=1` = healthy) and hides NaNs as fake codes (`ca=4`, `thal=0`). The 1,025-row copy is the same patients duplicated, which leaks test rows into training. If a Kaggle file is ever used, remap it to the UCI codes and dedupe first.
- **Target:** `target = (num > 0).astype(int)`. 1 = disease. Expect 139/303 positive (~46%).
- **Missing:** `ca` (4 rows) and `thal` (2 rows) only. Impute inside the pipeline with `most_frequent`.
- **Source codes:** `cp` 1–4 (4 = asymptomatic), `restecg` 0–2, `slope` 1–3, `ca` 0–3, `thal` 3/6/7.
- `validate()` must raise on unexpected codes, a wrong row count or missing columns. Don't coerce silently.

## Database (PostgreSQL + SQLAlchemy)

| Table | Written by | Purpose |
|---|---|---|
| `patients_raw` | `ingest.py` | 13 features + `num`, `source_site`, `ingested_at`. UCI values as-is; `?` → NULL |
| `model_runs` | `train.py` / `evaluate.py` | `run_id`, `model_name`, `params` JSONB, `threshold`, `cv_metrics` / `test_metrics` JSONB, `data_hash`, `artifact_path`, `lib_versions` JSONB, `is_active` |
| `predictions` | app, only if `LOG_PREDICTIONS=true` | `run_id` FK, the 13 inputs, `probability`, `label`, `created_at`. No identifiers, ever |

- Use SQLAlchemy 2.x style: `DeclarativeBase`, `Mapped[...]`, `mapped_column`, `select()`, `Session` as a context manager. No legacy `Query` API.
- Build the schema with `Base.metadata.create_all(engine)`. Add Alembic only if the schema starts changing.
- Read into pandas with `pd.read_sql(select(PatientRaw), engine)`.
- Use parameters or the ORM only. **Never f-string SQL.**
- `DATABASE_URL` comes from `.env` via python-dotenv. Never hard-code or commit credentials.
- Ingest is idempotent: if the row count and hash match, skip; otherwise truncate and reload inside one transaction.
- Store `data_hash` (SHA-256 of the sorted frame) on every `model_runs` row so each run is tied to its exact data.
- The model binary stays on disk (joblib) and only its path goes in the DB. Don't store pickles in Postgres.
- Keep one `is_active=true` run at a time, and switch it inside a transaction.

## Preprocessing (one ColumnTransformer, fit on train only)

| Group | Columns | Transform |
|---|---|---|
| Numeric | age, trestbps, chol, thalach, oldpeak, ca | SimpleImputer(most_frequent for ca) → StandardScaler |
| Nominal | cp, restecg, thal, slope | SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore") |
| Binary | sex, fbs, exang | passthrough |

## Hard rules

1. **Split before any fit.** Use a stratified 80/20 split with `random_state=42`. There is no `fit`/`fit_transform` on the full dataset anywhere.
2. **Preprocessing lives inside the saved `Pipeline`.** The app must never re-implement encoding or scaling.
3. **The test set is used once**, in `evaluate.py`. Model selection and threshold tuning use 5-fold stratified CV on train only.
4. **Threshold:** from `cross_val_predict(..., method="predict_proba")` on train, take the highest threshold with recall ≥ `RECALL_TARGET`. Store it in `metadata.json`.
5. **Imbalance:** `class_weight="balanced"` for LR and RF, and `scale_pos_weight=neg/pos` for XGBoost. No SMOTE; the classes are nearly balanced.
6. **Seed everything** (`SEED` from config) and log library versions into `metadata.json`.
7. **Logistic regression is the baseline to beat.** A tree model ships only if its CV recall beats LR by more than one CV std at comparable precision. Ties go to LR.
8. **Sanity alarm:** if test accuracy is above 0.92 or recall is 1.00, suspect leakage (duplicates, wrong file) and investigate before reporting.

## Evaluation output (required for all three models)

Recall, precision, F2, specificity, accuracy, ROC-AUC, PR-AUC, the false-negative count, and a confusion matrix at the chosen threshold. Recall and precision get 95% bootstrap CIs (1,000 resamples). Save the plots to `reports/figures/`: confusion matrices, ROC + PR overlays, recall-vs-threshold, and a calibration curve. Write `reports/model_comparison.md` with the table and a one-paragraph selection rationale. The test set is ~61 rows, so one miss is about 3.6 recall points. Say so in the report.

## Streamlit app rules

- Load the active `model_runs` row + joblib once with `@st.cache_resource`. No training at runtime. If Postgres is unreachable, fall back to `models/metadata.json` and disable logging.
- Use plain-language labels mapped to UCI codes. Read input bounds from `metadata.json`, not hard-coded values.
- `ca` and `thal` offer "Unknown", which sends NaN to the imputer.
- Output the probability %, the threshold, and the label **"Elevated risk — recommend clinical follow-up"** or **"Lower risk"**. Never say "has/does not have heart disease".
- Show the top 3 contributing features for the patient, plus an expander with model name, test recall and precision, and the confusion matrix.
- Persistent disclaimer: educational demo, 303 patients from a single clinic (1980s data, 68% male), not a diagnostic tool.
- Inputs are not stored unless `LOG_PREDICTIONS=true`, and then only to `predictions`, with no identifiers.

## Tests that must exist

- `test_data.py`: schema and value sets; the **label direction check** (mean `ca` and `oldpeak` higher for `target=1`); no duplicates after cleaning; stratification preserved.
- `test_db.py` (`@pytest.mark.db`, docker Postgres): ingest gives 303 rows and 6 NULLs; re-running ingest is a no-op; `model_runs` round-trips JSONB; only one active run.
- `test_features.py`: the preprocessor is fit on train only; the output has no NaN; column count is stable; unknown categories don't crash.
- `test_app_inference.py`: 3 fixed patient profiles through the app's inference function match `pipeline.predict_proba`; "Unknown" inputs work.

## Style

- Python 3.11+, type hints on public functions, docstrings on anything in `src/`.
- Keep functions pure where possible. Config lives in `config.py`, not scattered literals.
- Small, focused commits. Run `pytest` and `ruff` before each one.
- When a result looks surprisingly good, say so and check it. Don't celebrate it.

## Build order

M0 database + ingest → M1 data → M2 preprocessing → M3 models and comparison → M4 packaging → M5 Streamlit. Don't start a milestone until the previous one's tests pass.
