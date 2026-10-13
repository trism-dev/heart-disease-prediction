"""Heart disease risk demo. Loads the active run once; no training at runtime."""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURES, FIGURES_DIR  # noqa: E402
from src.inference import load_bundle, log_prediction, predict  # noqa: E402

st.set_page_config(page_title="Heart disease risk demo", page_icon="🫀", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.banner {background:#F6E3E1; border-left:4px solid #B3261E; padding:.6rem .9rem;
         border-radius:4px; font-size:.9rem; margin-bottom:1rem;}
.card {border-radius:10px; padding:1rem 1.2rem; margin-bottom:.8rem;}
.card.hi {background:#F6E3E1; border:1px solid #B3261E;}
.card.lo {background:#E3F0E6; border:1px solid #2E7D4F;}
.card h3 {margin:0 0 .2rem 0;}
.track {position:relative; height:14px; background:#E4DED5; border-radius:7px; margin:1.4rem 0 .3rem;}
.fill {height:14px; border-radius:7px;}
.mark {position:absolute; top:-6px; width:2px; height:26px; background:#1F1B16;}
.marklbl {position:absolute; top:-24px; font-size:.75rem; transform:translateX(-50%); white-space:nowrap;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def bundle():
    return load_bundle()


B = bundle()
META = B["meta"]
R = META["input_ranges"]
REF = META["reference_values"]
DS = META["dataset"]

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
SEX = {1: "Male", 0: "Female"}
CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "No symptoms"}
YN = {0: "No", 1: "Yes"}
ECG = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
CA = {0: "0", 1: "1", 2: "2", 3: "3", None: "Unknown"}
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect", None: "Unknown"}

PRESETS = {
    "Example 1: 63-year-old man": dict(
        age=63,
        sex=1,
        cp=1,
        trestbps=145,
        chol=233,
        fbs=1,
        restecg=2,
        thalach=150,
        exang=0,
        oldpeak=2.3,
        slope=3,
        ca=0,
        thal=6,
    ),  # fmt: skip
    "Example 2: 67-year-old man": dict(
        age=67,
        sex=1,
        cp=4,
        trestbps=160,
        chol=286,
        fbs=0,
        restecg=2,
        thalach=108,
        exang=1,
        oldpeak=1.5,
        slope=2,
        ca=3,
        thal=3,
    ),  # fmt: skip
    "Example 3: 41-year-old woman": dict(
        age=41,
        sex=0,
        cp=2,
        trestbps=130,
        chol=204,
        fbs=0,
        restecg=2,
        thalach=172,
        exang=0,
        oldpeak=1.4,
        slope=1,
        ca=0,
        thal=3,
    ),  # fmt: skip
}


def pct_text(p: float) -> str:
    """Avoid showing a misleading 0% / 100% for an uncalibrated score."""
    return ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{p:.0%}"


DEFAULTS = {k: (int(v) if k != "oldpeak" else float(v)) for k, v in REF.items()}


def load_values(values: dict) -> None:
    for k, v in values.items():
        st.session_state[f"in_{k}"] = v


for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"in_{k}", v)

st.title("Heart disease risk demo")
st.markdown(
    f'<div class="banner"><b>Educational demo, not a diagnostic tool.</b> Trained on {DS["rows"]} '
    f"patients from one US clinic (Cleveland Clinic, 1980s data, {DS['male_rate']:.0%} male).</div>",
    unsafe_allow_html=True,
)
if B["db_configured"] and not B["db_ok"]:
    st.info("Database unreachable: running from models/metadata.json, prediction logging is off.")

tab_predict, tab_about = st.tabs(["Predict", "About the model"])

with tab_predict:
    st.caption("Load an example from the training data, or enter your own values.")
    cols = st.columns(len(PRESETS) + 1)
    for c, (name, vals) in zip(cols, PRESETS.items()):
        c.button(name, on_click=load_values, args=(vals,), use_container_width=True)
    cols[-1].button("Reset to typical values", on_click=load_values, args=(DEFAULTS,),
                    use_container_width=True)  # fmt: skip

    left, right = st.columns([1, 1], gap="large")
    with left:
        a, b = st.columns(2)
        with a:
            st.number_input("Age (years)", int(R["age"][0]), int(R["age"][1]), key="in_age",
                            help="Training range 29–77.")  # fmt: skip
            st.selectbox("Sex", list(SEX), format_func=SEX.get, key="in_sex")
            st.selectbox("Chest pain type", list(CP), format_func=CP.get, key="in_cp",
                         help="'No symptoms' (asymptomatic) is the highest-risk group in this data.")  # fmt: skip
            st.number_input("Resting blood pressure (mm Hg)", int(R["trestbps"][0]),
                            int(R["trestbps"][1]), key="in_trestbps")  # fmt: skip
            st.number_input("Cholesterol (mg/dl)", int(R["chol"][0]), int(R["chol"][1]),
                            key="in_chol")  # fmt: skip
            st.selectbox("Fasting blood sugar > 120 mg/dl", list(YN), format_func=YN.get,
                         key="in_fbs")  # fmt: skip
            st.selectbox("Resting ECG", list(ECG), format_func=ECG.get, key="in_restecg")
        with b:
            st.number_input("Max heart rate (bpm)", int(R["thalach"][0]), int(R["thalach"][1]),
                            key="in_thalach")  # fmt: skip
            st.selectbox("Exercise-induced angina", list(YN), format_func=YN.get, key="in_exang")
            st.number_input("ST depression (oldpeak, mm)", float(R["oldpeak"][0]),
                            float(R["oldpeak"][1]), step=0.1, key="in_oldpeak",
                            help="ST depression induced by exercise relative to rest.")  # fmt: skip
            st.selectbox("ST slope", list(SLOPE), format_func=SLOPE.get, key="in_slope")
            st.selectbox("Vessels colored (0–3)", list(CA), format_func=CA.get, key="in_ca",
                         help="Major vessels seen by fluoroscopy. 'Unknown' is imputed.")  # fmt: skip
            st.selectbox("Thallium test", list(THAL), format_func=THAL.get, key="in_thal",
                         help="'Unknown' is imputed with the most common value.")  # fmt: skip

    inputs = {k: st.session_state[f"in_{k}"] for k in FEATURES}
    res = predict(B, inputs)

    with right:
        hi = res["elevated"]
        st.markdown(
            f'<div class="card {"hi" if hi else "lo"}"><h3>{res["label"]}</h3>'
            f"Model risk score <b>{pct_text(res['probability'])}</b> against a decision threshold of "
            f"<b>{res['threshold']:.0%}</b>.</div>",
            unsafe_allow_html=True,
        )
        pct, thr = res["probability"] * 100, res["threshold"] * 100
        colour = "#B3261E" if hi else "#2E7D4F"
        st.markdown(
            f'<div class="track"><div class="fill" style="width:{pct:.0f}%;background:{colour}"></div>'
            f'<div class="mark" style="left:{thr:.0f}%"></div>'
            f'<div class="marklbl" style="left:{thr:.0f}%">threshold</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "The threshold is set low on purpose: the model is tuned to miss as few sick patients "
            "as possible, which means more false alarms. The score is recalibrated to the "
            "Cleveland data (46% of patients had disease), so it only reads as a chance of "
            "disease for a similar population."
        )

        edge = [FRIENDLY[c] for c in ("age", "trestbps", "chol", "thalach", "oldpeak")
                if inputs[c] in R[c]]  # fmt: skip
        if edge:
            st.warning("At the edge of the training range: " + ", ".join(edge))

        st.subheader("What drives this score")
        contrib = sorted(res["contributions"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        ax.barh([FRIENDLY[f] for f, _ in contrib], [v for _, v in contrib],
                color=["#B3261E" if v > 0 else "#3B6EA5" for _, v in contrib])  # fmt: skip
        ax.axvline(0, color="#1F1B16", lw=0.8)
        ax.set_xlabel("Effect on log-odds vs a typical patient")
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=200)
        st.image(buf)
        plt.close(fig)
        names = [f"**{FRIENDLY[f]}** ({'raises' if v > 0 else 'lowers'})" for f, v in res["top"]]
        st.write("Biggest drivers: " + ", ".join(names) + ".")

    if st.button("Log this prediction", help="Only works when LOG_PREDICTIONS=true"):
        st.write("Logged." if log_prediction(B, inputs, res) else "Logging is off.")

with tab_about:
    m = B["test_metrics"]
    cv = META["cv_metrics"]["at_threshold"]
    st.markdown(
        f"**Model:** {B['model_name'].replace('_', ' ')}  \n"
        f"**Why this one:** {META['selection_reason']}."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Test recall", f"{m['recall']:.2f}",
              help=f"95% CI {m['recall_ci'][0]:.2f}–{m['recall_ci'][1]:.2f}")  # fmt: skip
    c2.metric("Test precision", f"{m['precision']:.2f}",
              help=f"95% CI {m['precision_ci'][0]:.2f}–{m['precision_ci'][1]:.2f}")  # fmt: skip
    c3.metric("Cross-validated precision", f"{cv['precision']['mean']:.2f}")
    st.write(
        "The test set has only about 61 patients (28 with disease), so one missed patient moves "
        "recall by about 3.6 points. Recall was lower for women than men in cross-validation "
        "(see the report), so the model should not be assumed to work equally well for both."
    )
    fig = FIGURES_DIR / "confusion_matrices.png"
    if fig.exists():
        st.image(str(fig), caption="Test-set confusion matrices, false negatives in red")
    st.markdown(
        "Full evaluation: `reports/model_comparison.md`. Data: UCI Heart Disease, CC BY 4.0."
    )
