# Heart Disease Prediction System — PRD

Sep 29, 2026 · @Tristan

## Overview

We will build a binary classifier on the UCI Heart Disease data. It flags patients likely to have heart disease, and it is tuned to catch sick patients first, with accuracy second. Three models get compared on the same split: logistic regression, random forest and XGBoost. The best one ships inside a Streamlit app where a user enters one patient's values and gets a risk score.

**Why recall comes first.** A false negative sends a sick patient home. A false positive costs one follow-up test. So the model is chosen and its decision threshold is set to hit a recall target, and precision is kept as high as possible under that target.

**Users**

- Primary: the builder, as a portfolio and learning project, plus reviewers reading the repo.
- Secondary: a non-technical person using the Streamlit form to try patient profiles.

**Success looks like**

- A reproducible pipeline: `python -m src.train` runs from the Postgres patient table to a saved model, with metrics logged to the database and no manual steps.
- A side-by-side comparison of all three models on recall, precision, F1, ROC-AUC and PR-AUC, with a confusion matrix for each.
- Test-set recall of at least 0.90 for the shipped model at its chosen threshold. This is a starting target; see Open questions.
- A Streamlit app that loads the saved pipeline and returns a probability, a label and the threshold used, in under 1 second.

This is an educational tool, not a medical device. It must never be presented as a diagnosis.

## Dataset

Use the original Cleveland subset of [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart-disease) (303 patients, 13 features plus the target, CC BY 4.0). Load it with the `ucimlrepo` package or from `processed.cleveland.data`. Do not take the popular Kaggle `heart.csv` at face value. Its target label is inverted and its missing values are hidden as fake category codes.

**What we verified in the files (2026-09-29)**

| Check | Original Cleveland ([mirror](https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv)) | Kaggle "Heart Disease UCI" 303-row copy ([mirror](https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv)) |
| --- | --- | --- |
| Rows | 303 | 303 (1 exact duplicate) |
| Missing values | 4 in `ca`, 2 in `thal` | 0 shown; they appear as `ca = 4` (5 rows) and `thal = 0` (2 rows), codes that do not exist in the source |
| Disease positive | 139 of 303 (46%) | `target = 1` on 165 rows |
| Label direction | 1–4 = disease, 0 = none | Inverted: `target = 1` rows have lower `ca`, `exang` and `oldpeak` and higher `thalach`, the healthy profile |
| `cp` coding | 1 typical angina … 4 asymptomatic | Re-mapped to 0–3 |

The 1,025-row Kaggle file ("Heart Disease Dataset") is widely reported to be the same \~303 patients repeated, which leaks test rows into training. We could not open it from here. If it is used, run `df.duplicated().sum()` first and drop duplicates before splitting.

**Features**

| Column | Meaning | Type | Values in source |
| --- | --- | --- | --- |
| `age` | Age | numeric | years |
| `sex` | Sex | binary | 1 male, 0 female |
| `cp` | Chest pain type | nominal | 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| `trestbps` | Resting blood pressure | numeric | mm Hg |
| `chol` | Serum cholesterol | numeric | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | binary | 1 yes, 0 no |
| `restecg` | Resting ECG | nominal | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| `thalach` | Max heart rate achieved | numeric | bpm |
| `exang` | Exercise-induced angina | binary | 1 yes, 0 no |
| `oldpeak` | ST depression, exercise vs rest | numeric | mm |
| `slope` | Slope of peak exercise ST segment | ordinal | 1 up, 2 flat, 3 down |
| `ca` | Major vessels colored by fluoroscopy | ordinal | 0–3 |
| `thal` | Thallium stress test | nominal | 3 normal, 6 fixed defect, 7 reversible defect |
| `num` → `target` | Diagnosis | target | 0 none; 1–4 disease → binarise to `target = (num > 0)` |

The target is roughly balanced (46% positive), so SMOTE is not needed. Use `class_weight` and threshold tuning instead.

## Data pipeline

All preprocessing lives inside one scikit-learn `Pipeline` that is fit on training data only and saved with the model. The Streamlit app calls that same object, so training and serving can never disagree.

1. **Ingest, load and validate.** `src/ingest.py` loads the UCI file into the Postgres `patients_raw` table once, storing `?` as NULL. `src/data.py` reads that table through SQLAlchemy (`pd.read_sql(select(PatientRaw), engine)`). Assert 303 rows, 14 columns and the value sets in the Features table. Fail loudly on any unknown code.
2. **Binarise the target.** `target = (num > 0).astype(int)`. Drop `num`.
3. **Deduplicate.** Drop exact duplicate rows before splitting, and log how many were removed.
4. **Split first.** Stratified 80/20 train/test split, `random_state=42`. The test set is touched once, at final evaluation. Model selection uses 5-fold stratified cross-validation on the training set.
5. **Missing values.** Only `ca` (4) and `thal` (2) are missing, about 2% of rows. Impute with `SimpleImputer(strategy="most_frequent")` inside the pipeline. Report the counts in the EDA notebook. Dropping the 6 rows is an acceptable alternative, but the imputer stays so the app can handle a blank field.
6. **Encode categoricals.** `OneHotEncoder(handle_unknown="ignore")` on `cp`, `restecg`, `thal` and `slope`. Pass `sex`, `fbs` and `exang` through as 0/1. Keep `ca` as an ordinal integer.
7. **Scale numerics.** `StandardScaler` on `age`, `trestbps`, `chol`, `thalach`, `oldpeak` and `ca`. Logistic regression needs it. The tree models don't, but one shared preprocessor keeps the comparison fair.

**Leakage rules**

- No `fit` or `fit_transform` on the full dataset, ever. The imputer, encoder and scaler see training folds only.
- No feature selection or outlier removal before the split.
- Near-duplicate patients across the split: flag them in EDA and do not fix them silently.

**EDA deliverable** (`notebooks/01_eda.ipynb`): the missing-value table, class balance, a histogram per numeric feature by target, and a correlation heatmap. Also call out `chol = 0` or `trestbps = 0` if they appear. The Cleveland file has none, but the other UCI sites do.

## Modeling

All three models share the preprocessor and CV folds. Each is tuned with `RandomizedSearchCV` using 5-fold stratified CV and `scoring="recall"`, with `refit` on the best recall that also keeps precision at or above 0.70. On 242 training rows a large grid overfits the CV, so searches are capped at 50 iterations.

| Model | Role | Search space (starting point) | Imbalance handling |
| --- | --- | --- | --- |
| Logistic regression | Baseline; interpretable coefficients | `C` in logspace(-3, 2); `penalty` l1 or l2; `solver="liblinear"` | `class_weight="balanced"` |
| Random forest | Non-linear, low tuning risk | `n_estimators` 200–800; `max_depth` 3–10 or None; `min_samples_leaf` 1–10; `max_features` sqrt or 0.5 | `class_weight="balanced"` |
| XGBoost (`XGBClassifier`) | Gradient boosting | `n_estimators` 100–600; `max_depth` 2–6; `learning_rate` 0.01–0.3; `subsample` and `colsample_bytree` 0.6–1.0; `reg_lambda` 0.1–10 | `scale_pos_weight = neg / pos` from the training split |

**Rules**

- Logistic regression is the bar to beat. A complex model ships only if its CV recall beats the baseline by more than one CV standard deviation at comparable precision. Otherwise ship logistic regression, because it's simpler and explainable.
- Fix `random_state=42` everywhere. Log library versions with every run.
- Explainability: logistic regression coefficients as odds ratios, and permutation importance for the tree models. SHAP is optional.
- Calibration: check a reliability curve for the shipped model. Wrap it in `CalibratedClassifierCV` if the probabilities are off, because the app shows a probability, not just a label.

## Evaluation

The shipped model is the one with the best recall at a precision floor, not the best accuracy. The decision threshold is tuned. Nobody should assume 0.5.

**Metrics reported for every model** (CV mean ± std, then test)

- Recall (sensitivity): the primary metric. Share of sick patients caught.
- Precision: share of flagged patients who are sick.
- F2 score: weights recall twice as much as precision. Used to break ties.
- Specificity, accuracy, ROC-AUC, PR-AUC.
- Confusion matrix at the chosen threshold, with false negatives highlighted.

**Threshold selection**

1. Collect out-of-fold probabilities on the training set with `cross_val_predict(method="predict_proba")`.
2. Pick the highest threshold where recall is at least 0.90. A higher threshold means fewer false alarms at the same recall.
3. Save the threshold with the model in `models/metadata.json`.
4. Apply it once to the test set. Never tune on test.

**Small-sample honesty.** The test set holds about 61 patients, about 28 of them sick. One missed patient moves recall by about 3.6 points. Report 95% bootstrap confidence intervals (1,000 resamples) for recall and precision. Treat differences inside the intervals as ties.

**Comparison output** (`reports/model_comparison.md`, generated by the evaluate script; values stay blank until the run)

| Model | Threshold | Recall | Precision | F2 | Specificity | ROC-AUC | PR-AUC | False negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression |  |  |  |  |  |  |  |  |
| Random forest |  |  |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |  |  |

**Plots** saved to `reports/figures/`: one confusion matrix per model, overlaid ROC and precision-recall curves, a recall-vs-threshold curve with the chosen point marked, and the calibration curve of the shipped model.

## Streamlit app

The app is a single page. A user fills a 13-field form in plain language, clicks Predict, and gets a probability, a label and the reasons behind it. It loads `models/pipeline.joblib` and `models/metadata.json` once with `@st.cache_resource`, and does no training at runtime.

**Inputs.** Plain-language labels, with the source codes mapped behind the scenes. Ranges are clamped to what the training data covers.

| Field | Widget | Allowed range / options |
| --- | --- | --- |
| Age | number input | 29–77 years |
| Sex | radio | Male, Female |
| Chest pain type | select | Typical angina, Atypical angina, Non-anginal pain, No symptoms (asymptomatic) |
| Resting blood pressure | number input | 94–200 mm Hg |
| Cholesterol | number input | 126–564 mg/dl |
| Fasting blood sugar > 120 mg/dl | toggle | Yes, No |
| Resting ECG | select | Normal, ST-T abnormality, LV hypertrophy |
| Max heart rate | number input | 71–202 bpm |
| Exercise-induced angina | toggle | Yes, No |
| ST depression (oldpeak) | number input | 0.0–6.2, step 0.1 |
| ST slope | select | Upsloping, Flat, Downsloping |
| Vessels colored (ca) | select | 0, 1, 2, 3, Unknown |
| Thallium test (thal) | select | Normal, Fixed defect, Reversible defect, Unknown |

"Unknown" sends NaN to the pipeline's imputer. The ranges are the Cleveland min–max. The app reads them from `metadata.json`, so they are never hard-coded twice.

**Outputs**

- The risk probability as a percentage, with the model's threshold shown next to it.
- A label: "Elevated risk — recommend clinical follow-up" or "Lower risk". Never "has" or "does not have" disease.
- The top 3 contributing features for this patient: LR coefficient × value, or SHAP for tree models.
- An expander with the model name, test recall, precision, the confusion matrix image and the dataset size.

**Guardrails**

- A persistent banner: educational demo trained on 303 patients from one US clinic (Cleveland Clinic, data from the 1980s); not a diagnostic tool.
- A warning when any input sits at the edge of the training range.
- Predictions are not stored by default. Setting LOG\_PREDICTIONS=true writes inputs and output to the predictions table, never with names or identifiers.

## Architecture

One training script produces one saved pipeline, and the app only loads it. The held-out 20% is used once, after the model and threshold are fixed.

&#91;embedded content: training-to-serving flow · 8 steps\]

**Stack**

| Layer | Technology | Used for |
| --- | --- | --- |
| Language | Python 3.11+ | Everything |
| Data wrangling | pandas | EDA, cleaning, feature frames |
| Modeling | XGBoost ≥ 2.0, scikit-learn ≥ 1.4 | Gradient boosting; scikit-learn supplies logistic regression, random forest, pipelines, CV and metrics |
| Database | PostgreSQL 16 | Source-of-truth patient table, training-run history, optional prediction log |
| DB access | SQLAlchemy 2.x + psycopg 3 | ORM models, engine and sessions; `pd.read_sql` through the engine |
| App | Streamlit | Patient input form and prediction |
| Support | joblib, matplotlib, python-dotenv, pytest, ruff | Artifacts, plots, config, tests, lint |

scikit-learn isn't on the stated technology list, but the plan needs it. Logistic regression, random forest, the preprocessing pipeline and every metric come from it.

### Data layer

PostgreSQL holds the data and the run history. The model file itself stays on disk as `models/pipeline.joblib`, and its path is recorded in the database. The connection comes from `DATABASE_URL` in `.env` (`postgresql+psycopg://user:pass@localhost:5432/heart`). A local database runs from `docker-compose.yml` (`postgres:16`).

| Table | Written by | Key columns | Notes |
| --- | --- | --- | --- |
| `patients_raw` | `src/ingest.py`, once | `id` PK, the 13 features, `num`, `source_site`, `ingested_at` | Values exactly as in UCI; `?` stored as NULL. Ingest is idempotent: it truncates and reloads, or skips if row count and hash already match |
| `model_runs` | `src/train.py`, `src/evaluate.py` | `run_id` PK, `created_at`, `model_name`, `params` JSONB, `threshold`, `cv_metrics` JSONB, `test_metrics` JSONB, `data_hash`, `artifact_path`, `lib_versions` JSONB, `is_active` | One row per model per training run. The app loads the run where `is_active` is true |
| `predictions` | Streamlit app, only when `LOG_PREDICTIONS=true` | `id` PK, `created_at`, `run_id` FK, the 13 inputs, `probability`, `label` | Off by default. No names or other identifiers, ever |

**Rules**

- Training reads from `patients_raw` through SQLAlchemy and never from a CSV. `data_hash` (SHA-256 of the sorted frame) ties every run to the exact rows it saw.
- Schema is defined once as SQLAlchemy ORM models in `src/db.py`, and `create_all()` builds it. Alembic is optional unless the schema starts changing.
- Queries use parameters or the ORM, never f-string SQL.
- The app works without a database: if Postgres is unreachable, it falls back to `models/metadata.json` and turns prediction logging off.

```
heart-disease-prediction/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docker-compose.yml       # postgres:16 for local dev
├── .env.example             # DATABASE_URL, LOG_PREDICTIONS=false
├── data/
│   └── raw/                 # downloaded UCI file, git-ignored
├── notebooks/
│   └── 01_eda.ipynb         # reads from Postgres via src.data
├── src/
│   ├── config.py            # paths, seed, column lists, recall target, env loading
│   ├── db.py                # engine, session factory, ORM models (3 tables)
│   ├── ingest.py            # UCI file -> patients_raw (idempotent)
│   ├── data.py              # read patients_raw, validate, binarise, dedupe, split
│   ├── features.py          # build_preprocessor() -> ColumnTransformer
│   ├── models.py            # model factories + search spaces
│   ├── train.py             # CV tuning, threshold, save joblib, write model_runs
│   └── evaluate.py          # test metrics, plots, report, update model_runs
├── models/
│   ├── pipeline.joblib
│   └── metadata.json        # offline fallback copy of the active run
├── reports/
│   ├── model_comparison.md
│   └── figures/
├── app/
│   └── streamlit_app.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_db.py           # @pytest.mark.db, needs the docker Postgres
    └── test_app_inference.py
```

## Milestones and acceptance criteria

Six milestones, built in order. Each one is done when its boxes are ticked, and none starts before the previous one passes.

**M0 — Database**

- [ ] `docker compose up` starts Postgres 16, and `.env.example` documents `DATABASE_URL` and `LOG_PREDICTIONS`.
- [ ] `src/db.py` defines `patients_raw`, `model_runs` and `predictions` as SQLAlchemy 2.x ORM models, and `create_all()` builds them.
- [ ] `python -m src.ingest` loads 303 rows with 6 NULLs (4 `ca`, 2 `thal`). Running it twice leaves 303 rows.

**M1 — Data foundation**

- [ ] `src/data.py` loads the raw file, validates the schema and value sets, binarises the target and drops duplicates.
- [ ] Tests fail on an inverted label (they assert disease rows have mean `ca` > healthy rows), on unknown codes, and on a row-count mismatch.
- [ ] The EDA notebook shows missing values, class balance and feature distributions by target.

**M2 — Preprocessing**

- [ ] `build_preprocessor()` returns a `ColumnTransformer` covering impute, one-hot and scale.
- [ ] A test proves the preprocessor is fit on train only, and that the output has no NaN and a fixed column count.

**M3 — Models and comparison**

- [ ] Logistic regression, random forest and XGBoost are tuned with the same folds and seed.
- [ ] Thresholds are chosen on out-of-fold predictions.
- [ ] `reports/model_comparison.md` has the full metric table, bootstrap CIs and false-negative counts, and all plots are saved.
- [ ] The model-selection decision is written up in the report, with its reason.

**M4 — Packaging**

- [ ] `models/pipeline.joblib` and `metadata.json` hold the model, threshold, metrics, input ranges and library versions.
- [ ] `python -m src.train && python -m src.evaluate` reproduces the metrics exactly on a clean clone.

**M5 — Streamlit app**

- [ ] `streamlit run app/streamlit_app.py` serves the form, prediction, threshold, top contributing features and disclaimer.
- [ ] A test feeds 3 fixed patient profiles through the app's inference function and matches the saved pipeline's output.
- [ ] "Unknown" for `ca`/`thal` works end to end.
- [ ] A prediction returns in under 1 second locally.

## Risks, non-goals, open questions

**Risks**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Wrong Kaggle file used (inverted label, fake codes, duplicates) | The model learns the opposite of reality, or scores look inflated | Load from UCI. Schema and direction tests in M1 fail the build otherwise |
| Tiny test set (\~61 rows) | Model ranking is noise | CV mean ± std for selection, bootstrap CIs on test, and ties go to logistic regression |
| Very high published accuracy (95%+) on this data | Usually a sign of the duplicated 1,025-row file leaking | Any test accuracy above 0.92 triggers a leakage check before it is reported |
| One clinic, 1980s patients, 68% male | Poor generalisation to other groups | State it in the app and README. Optional: evaluate on the Hungarian/Swiss/VA sites as an external test |
| Users read the output as a diagnosis | Harm | Risk wording, never diagnosis wording; a persistent disclaimer; no storage |

PostgreSQL is far more than a 303-row dataset needs. It earns its place through reproducible run history (`model_runs`) and the optional prediction log, not through data volume.

**Non-goals:** clinical deployment, user accounts, storing identifiable patient data, a REST API, retraining from the app, and multi-class severity (`num` 1–4).

**Open questions**

- Is 0.90 the right recall target? 0.95 is possible but will likely push precision toward 0.60. The recall-vs-threshold curve from M3 should settle it.
- Should the four-site, 920-row UCI data be used for training (more rows, heavy missingness in `ca`, `thal` and `slope`), or only for external validation?
- Where will the app be hosted: Streamlit Community Cloud, or local only?

## Sources

- [UCI Machine Learning Repository — Heart Disease (dataset 45)](https://archive.ics.uci.edu/dataset/45/heart-disease)
- [Original Cleveland file, ISLR mirror (Heart.csv)](https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv): used to verify missing values and class balance
- [Kaggle "Heart Disease UCI" 303-row copy, GitHub mirror](https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv): used to verify the label inversion and recoded NaNs
