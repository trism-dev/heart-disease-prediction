"""Test-set metrics (used once), bootstrap CIs, plots, report; sets model_runs.is_active + packages the shipped model."""

import json
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sqlalchemy import select, update

from src.config import (
    FEATURES,
    FIGURES_DIR,
    METADATA_PATH,
    MODELS_DIR,
    N_BOOT,
    PIPELINE_PATH,
    REPORTS_DIR,
    SEED,
)
from src.data import data_hash, load_clean, split
from src.db import ModelRun, get_engine, get_session_factory
from src.models import MODEL_NAMES
from src.train import lib_versions

LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}


def metrics_at(y, proba, thr: float) -> dict:
    pred = proba >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(y, proba, thr: float, n: int = N_BOOT) -> dict:
    """95% percentile CIs for recall and precision over n resamples of the test set."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    rec, prec = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        p = proba[i] >= thr
        rec.append(recall_score(y[i], p))
        prec.append(precision_score(y[i], p, zero_division=0))
    return {
        "recall_ci": [float(x) for x in np.percentile(rec, [2.5, 97.5])],
        "precision_ci": [float(x) for x in np.percentile(prec, [2.5, 97.5])],
    }


def plot_confusions(res):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, MODEL_NAMES):
        m = res[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            fn = (i, j) == (1, 0)
            ax.text(
                j,
                i,
                v,
                ha="center",
                va="center",
                fontsize=16,
                color="crimson" if fn else "black",
                fontweight="bold" if fn else None,
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Pred lower", "Pred elevated"],
            yticklabels=["Actual none", "Actual disease"],
            title=f"{LABELS[name]} (thr {m['threshold']:.2f})",
        )
    fig.suptitle("Test-set confusion matrices (false negatives in red)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_roc_pr(y, probs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y, probs[name])
        a.plot(fpr, tpr, label=f"{LABELS[name]} (AUC {roc_auc_score(y, probs[name]):.3f})")
        p, r, _ = precision_recall_curve(y, probs[name])
        b.plot(r, p, label=f"{LABELS[name]} (AP {average_precision_score(y, probs[name]):.3f})")
    a.plot([0, 1], [0, 1], "k:", lw=1)
    a.set(xlabel="False positive rate", ylabel="Recall", title="ROC (test)")
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    a.legend()
    b.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr.png", dpi=130)
    plt.close(fig)


def plot_recall_threshold(oof, y_train, probs, y_test, thresholds, shipped):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    grid = np.linspace(0, 1, 201)
    for ax, name in zip(axes, MODEL_NAMES):
        ax.plot(grid, [recall_score(y_train, oof[name] >= t) for t in grid], label="train OOF")
        ax.plot(grid, [recall_score(y_test, probs[name] >= t) for t in grid], "--", label="test")
        ax.axvline(thresholds[name], color="crimson", lw=1)
        ax.axhline(0.9, color="gray", ls=":")
        ax.set(title=LABELS[name] + (" (shipped)" if name == shipped else ""), xlabel="Threshold")
    axes[0].set_ylabel("Recall")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "recall_vs_threshold.png", dpi=130)
    plt.close(fig)


def plot_calibration(y, proba, name):
    frac, mean = calibration_curve(y, proba, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    ax.plot(mean, frac, "o-", label=LABELS[name])
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction with disease",
        title="Calibration (test, 5 quantile bins)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=130)
    plt.close(fig)


def main() -> None:
    engine = get_engine()
    sel = json.loads((MODELS_DIR / "selection.json").read_text())
    shipped = sel["selected"]
    oof_blob = joblib.load(MODELS_DIR / "oof.joblib")
    df = load_clean(engine)
    _, X_test, _, y_test = split(df)
    y = y_test.to_numpy()

    Session = get_session_factory(engine)
    with Session() as s:
        runs = {
            r.model_name: r
            for r in s.scalars(select(ModelRun).where(ModelRun.run_id.in_(sel["run_ids"].values())))
        }
    if data_hash(df) != sel["data_hash"]:
        raise RuntimeError("patients_raw changed since training; re-run src.train")

    res, probs, thresholds = {}, {}, {}
    for name in MODEL_NAMES:
        pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
        probs[name] = pipe.predict_proba(X_test)[:, 1]
        thresholds[name] = runs[name].threshold
        res[name] = {
            **metrics_at(y, probs[name], thresholds[name]),
            **bootstrap_ci(y, probs[name], thresholds[name]),
        }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusions(res)
    plot_roc_pr(y, probs)
    plot_recall_threshold(oof_blob["oof"], oof_blob["y_train"], probs, y, thresholds, shipped)
    plot_calibration(y, probs[shipped], shipped)

    # DB: write test metrics, switch the active run in one transaction
    with Session.begin() as s:
        for name, r in runs.items():
            s.execute(
                update(ModelRun)
                .where(ModelRun.run_id == r.run_id)
                .values(test_metrics=res[name], is_active=(name == shipped))
            )
        s.execute(
            update(ModelRun)
            .where(ModelRun.run_id.notin_(sel["run_ids"].values()))
            .values(is_active=False)
        )

    # Packaging
    shutil.copyfile(MODELS_DIR / f"{shipped}.joblib", PIPELINE_PATH)
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES}
    ref = {
        c: float(df[c].median())
        if c in ("age", "trestbps", "chol", "thalach", "oldpeak")
        else float(df[c].mode().iloc[0])
        for c in FEATURES
    }
    meta = {
        "model_name": shipped,
        "run_id": runs[shipped].run_id,
        "threshold": thresholds[shipped],
        "selection_reason": sel["reason"],
        "cv_metrics": runs[shipped].cv_metrics,
        "test_metrics": res[shipped],
        "input_ranges": ranges,
        "reference_values": ref,
        "dataset": {
            "rows": len(df),
            "positive_rate": float(df["target"].mean()),
            "male_rate": float(df["sex"].mean()),
            "data_hash": sel["data_hash"],
        },
        "lib_versions": lib_versions(),
        "seed": SEED,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))

    write_report(res, runs, shipped, sel, len(y), int(y.sum()))
    m = res[shipped]
    print(
        f"shipped: {shipped} | test recall {m['recall']:.3f} precision {m['precision']:.3f} "
        f"accuracy {m['accuracy']:.3f} FN {m['fn']}"
    )
    if m["accuracy"] > 0.92 or m["recall"] == 1.0:
        print(
            "SANITY ALARM: accuracy > 0.92 or recall == 1.00 -- investigate leakage before reporting"
        )


def write_report(res, runs, shipped, sel, n_test, n_pos) -> None:
    rows = []
    for name in MODEL_NAMES:
        m = res[name]
        rows.append(
            f"| {LABELS[name]} | {m['threshold']:.3f} | {m['recall']:.3f} "
            f"({m['recall_ci'][0]:.2f}–{m['recall_ci'][1]:.2f}) | {m['precision']:.3f} "
            f"({m['precision_ci'][0]:.2f}–{m['precision_ci'][1]:.2f}) | {m['f2']:.3f} | "
            f"{m['specificity']:.3f} | {m['accuracy']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | "
            f"{m['fn']} |"
        )
    cv = []
    for name in MODEL_NAMES:
        c = runs[name].cv_metrics
        s, t = c["search"], c["at_threshold"]
        cv.append(
            f"| {LABELS[name]} | {s['recall']['mean']:.3f} ± {s['recall']['std']:.3f} | "
            f"{s['precision']['mean']:.3f} ± {s['precision']['std']:.3f} | "
            f"{t['recall']['mean']:.3f} ± {t['recall']['std']:.3f} | "
            f"{t['precision']['mean']:.3f} ± {t['precision']['std']:.3f} | {c['oof_precision']:.3f} |"
        )
    m = res[shipped]
    perfect = "".join(
        f"Note: {LABELS[n]} reaches recall 1.00 on test at threshold {res[n]['threshold']:.3f} with "
        f"specificity {res[n]['specificity']:.3f}. That comes from flagging many healthy patients, "
        "not from leakage (accuracy stays low).\n\n"
        for n in MODEL_NAMES
        if res[n]["recall"] == 1.0 and n != shipped
    )
    md = f"""# Model comparison

Generated by `python -m src.evaluate`. Test set: {n_test} patients, {n_pos} with disease.
**One missed patient moves recall by about {100 / n_pos:.1f} points**, so differences inside the
95% bootstrap intervals (1,000 resamples) are ties.

## Test set at each model's chosen threshold

| Model | Threshold | Recall (95% CI) | Precision (95% CI) | F2 | Specificity | Accuracy | ROC-AUC | PR-AUC | False negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Cross-validation on train (5-fold stratified)

`Search` columns are the tuned model at the default 0.5 cut-off (what the selection rule uses).
`At threshold` columns apply the OOF-chosen threshold, so recall sits near 0.90 by construction;
the discriminating number there is precision.

| Model | Search recall | Search precision | Recall at threshold | Precision at threshold | OOF precision |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(cv)}

## Selection

**Shipped: {LABELS[shipped]}.** {sel["reason"]}. Rule: a tree model ships only if its CV recall
beats logistic regression by more than one CV standard deviation at comparable precision (within
0.05); ties go to LR. The shipped model catches {m["tp"]} of {n_pos} sick test patients
(recall {m["recall"]:.3f}, {m["fn"]} missed) with precision {m["precision"]:.3f} and
{m["fp"]} false alarms.

Thresholds are the highest cut-off with out-of-fold train recall ≥ 0.90; the test set was used once,
only to report. Class weighting (`balanced`) inflates predicted probabilities, so the app's
percentage is a risk score, not a calibrated prevalence estimate (see `figures/calibration.png`).

Sanity check: test accuracy {m["accuracy"]:.3f}, recall {m["recall"]:.3f}
({"ALARM: investigate leakage" if m["accuracy"] > 0.92 or m["recall"] == 1.0 else "inside the plausible range, no leakage alarm"}).

{perfect}## Figures

`figures/confusion_matrices.png`, `figures/roc_pr.png`, `figures/recall_vs_threshold.png`,
`figures/calibration.png`.
"""
    (REPORTS_DIR / "model_comparison.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
