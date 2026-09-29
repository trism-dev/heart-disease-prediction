# Heart Disease Prediction System

Binary classifier on the UCI Heart Disease (Cleveland) data, tuned for **recall first**: a missed sick patient is
worse than a false alarm. Logistic regression, random forest and XGBoost are compared on one split; the winner
is served in a Streamlit form. PostgreSQL stores patient rows and training-run history.

**Educational demo. 303 patients, one clinic, 1980s data, 68% male. Not a diagnostic tool.**

## Run it

```bash
py -3.12 -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d          # postgres:16 on host port 5433
python -m src.ingest          # data/raw file -> patients_raw (idempotent; downloads the file if absent)
python -m src.train           # ~7 min: CV-tune 3 models, OOF thresholds, model_runs rows
python -m src.evaluate        # test metrics, plots, reports/model_comparison.md, activates the winner
streamlit run app/streamlit_app.py
pytest -q                     # -m "not db" skips tests that need Postgres
```

Host port is 5433 because 5432 is often taken by another local Postgres. Change it in `docker-compose.yml`
and `.env` together if you prefer.

## Results

See [reports/model_comparison.md](reports/model_comparison.md). The test set is ~61 rows, so one miss is
~3.6 recall points; read differences inside the bootstrap CIs as ties.

## Notes

- Data: UCI file only. The common Kaggle `heart.csv` has an inverted target and fake missing-value codes;
  `validate()` and the label-direction test reject that.
- The shipped model's percentage is a class-weighted risk score, not a calibrated probability.
- Prediction logging is off unless `LOG_PREDICTIONS=true`, and never stores identifiers.
