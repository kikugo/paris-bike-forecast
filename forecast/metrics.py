"""Point-forecast metrics over (counter, hour) pairs with a known actual."""

import numpy as np

DECIMALS = 4


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in percent (0 to 200). Hours where both are 0 count as a perfect 0.

    Plain MAPE is undefined here: counts hit zero at night.
    """
    denom = np.abs(y) + np.abs(yhat)
    terms = np.divide(2 * np.abs(y - yhat), denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(100 * np.mean(terms))


def score(y, yhat) -> dict:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return {
        "mae": round(mae(y, yhat), DECIMALS),
        "rmse": round(rmse(y, yhat), DECIMALS),
        "smape": round(smape(y, yhat), DECIMALS),
        "n": len(y),
    }


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_score(y, lower, upper, quantiles: dict[float, np.ndarray]) -> dict:
    """80% interval coverage and width, and mean pinball loss over the given quantiles."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return {
        "coverage_80": round(float(np.mean((y >= lower) & (y <= upper))), DECIMALS),
        "interval_width": round(float(np.mean(upper - lower)), DECIMALS),
        "pinball": round(
            float(np.mean([pinball(y, np.asarray(p, dtype=float), q) for q, p in quantiles.items()])), DECIMALS
        ),
    }
