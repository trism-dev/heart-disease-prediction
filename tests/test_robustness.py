import numpy as np

from src.robustness import apply_platt, choose_threshold, fit_platt, target_table


def _toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.25 * y + rng.random(300) * 0.75, 0, 1)
    return y, p


def test_threshold_meets_recall_target_and_is_highest():
    y, p = _toy()
    t = choose_threshold(y, p, 0.90)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.90
    higher = p[p > t]
    if len(higher):
        t2 = higher.min()
        assert ((p >= t2) & (y == 1)).sum() / y.sum() < 0.90


def test_platt_is_monotone_so_decisions_do_not_change():
    y, p = _toy()
    platt = fit_platt(y, p)
    q = apply_platt(p, platt)
    t = choose_threshold(y, p)
    assert ((p >= t) == (q >= apply_platt(t, platt))).all()
    assert (np.diff(q[np.argsort(p)]) >= 0).all()


def test_higher_recall_target_never_raises_threshold():
    y, p = _toy()
    tt = target_table(y, p)
    assert tt["0.95"]["threshold"] <= tt["0.90"]["threshold"] <= tt["0.85"]["threshold"]
