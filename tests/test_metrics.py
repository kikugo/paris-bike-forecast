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
