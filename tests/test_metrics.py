import numpy as np
import pytest

from forecast import metrics


def test_hand_computed_values():
    y = np.array([0.0, 10.0, 20.0])
    yhat = np.array([0.0, 12.0, 14.0])
    s = metrics.score(y, yhat)
    assert s["mae"] == pytest.approx(8 / 3, abs=1e-4)
    assert s["rmse"] == pytest.approx(np.sqrt(40 / 3), abs=1e-4)
    # sMAPE terms: 0 (0/0), 2*2/22, 2*6/34
    assert s["smape"] == pytest.approx(100 * (0 + 4 / 22 + 12 / 34) / 3, abs=1e-4)
    assert s["n"] == 3


def test_smape_bounds():
    assert metrics.smape(np.array([0.0]), np.array([5.0])) == 200
    assert metrics.smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0


def test_interval_score():
    y = np.array([5.0, 10.0, 20.0, 0.0])
    lo = np.array([4.0, 11.0, 10.0, 0.0])
    hi = np.array([6.0, 12.0, 30.0, 1.0])
    s = metrics.interval_score(y, lo, hi, {0.1: lo, 0.9: hi})
    assert s["coverage_80"] == 0.75  # the second actual is below its interval
    assert s["interval_width"] == pytest.approx((2 + 1 + 20 + 1) / 4)
    # pinball at 0.1 (y - lo = 1, -1, 10, 0): 0.1, 0.9, 1.0, 0 -> 0.5
    # pinball at 0.9 (y - hi = -1, -2, -10, -1): 0.1, 0.2, 1.0, 0.1 -> 0.35
    assert s["pinball"] == pytest.approx((0.5 + 0.35) / 2)
