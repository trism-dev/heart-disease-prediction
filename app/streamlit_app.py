"""Heart disease risk demo. Loads the active run once; no training at runtime."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FIGURES_DIR
from src.inference import load_bundle, log_prediction, predict

st.set_page_config(page_title="Heart disease risk demo", page_icon="🫀")


@st.cache_resource
def bundle():
    return load_bundle()


B = bundle()
R = B["meta"]["input_ranges"]
ds = B["meta"]["dataset"]

FRIENDLY = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Vessels colored",
    "thal": "Thallium test",
}

st.title("🫀 Heart disease risk demo")
st.warning(
    f"Educational demo trained on {ds['rows']} patients from one US clinic (Cleveland Clinic, "
    f"1980s data, {ds['male_rate']:.0%} male). **Not a diagnostic tool.**"
)
if not B["db_ok"]:
    st.info("Database unreachable: running from models/metadata.json, prediction logging is off.")


def opt(label, mapping, key=None):
    return mapping[st.selectbox(label, list(mapping), key=key)]


with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", int(R["age"][0]), int(R["age"][1]), 55)
        sex = opt("Sex", {"Male": 1, "Female": 0})
        cp = opt(
            "Chest pain type",
            {
                "Typical angina": 1,
                "Atypical angina": 2,
                "Non-anginal pain": 3,
                "No symptoms (asymptomatic)": 4,
            },
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", int(R["trestbps"][0]), int(R["trestbps"][1]), 130
        )
        chol = st.number_input("Cholesterol (mg/dl)", int(R["chol"][0]), int(R["chol"][1]), 240)
        fbs = opt("Fasting blood sugar > 120 mg/dl", {"No": 0, "Yes": 1})
        restecg = opt("Resting ECG", {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2})
    with c2:
        thalach = st.number_input(
            "Max heart rate (bpm)", int(R["thalach"][0]), int(R["thalach"][1]), 150
        )
        exang = opt("Exercise-induced angina", {"No": 0, "Yes": 1})
        oldpeak = st.number_input(
            "ST depression (oldpeak, mm)", R["oldpeak"][0], R["oldpeak"][1], 1.0, step=0.1
        )
        slope = opt("ST slope", {"Upsloping": 1, "Flat": 2, "Downsloping": 3})
        ca = opt("Vessels colored (ca)", {"0": 0, "1": 1, "2": 2, "3": 3, "Unknown": None})
        thal = opt(
            "Thallium test (thal)",
            {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7, "Unknown": None},
        )
    go = st.form_submit_button("Predict")

if go:
    inputs = dict(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )
    res = predict(B, inputs)
    log_prediction(B, inputs, res)
    edge = [
        FRIENDLY[c] for c in ("age", "trestbps", "chol", "thalach", "oldpeak") if inputs[c] in R[c]
    ]
    if edge:
        st.warning("At the edge of the training range: " + ", ".join(edge))
    (st.error if res["elevated"] else st.success)(res["label"])
    a, b = st.columns(2)
    a.metric("Model risk score", f"{res['probability']:.0%}")
    b.metric("Decision threshold", f"{res['threshold']:.0%}")
    st.caption(
        "The score comes from a class-weighted model, so it is a ranking signal, "
        "not a calibrated probability."
    )
    st.subheader("Top 3 contributing features")
    for f, v in res["top"]:
        st.write(
            f"**{FRIENDLY.get(f, f)}**: {'raises' if v > 0 else 'lowers'} the score ({v:+.2f})"
        )

with st.expander("About the model"):
    m = B["test_metrics"]
    st.write(
        f"Model: **{B['model_name'].replace('_', ' ')}** · test recall {m['recall']:.2f} · "
        f"precision {m['precision']:.2f} · {ds['rows']} patients. The test set is about 61 "
        "patients, so these numbers are noisy."
    )
    fig = FIGURES_DIR / "confusion_matrices.png"
    if fig.exists():
        st.image(str(fig))
