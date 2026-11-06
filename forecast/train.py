"""Log the LightGBM backtest to MLflow, fit the final model, and promote it only if it wins.

Reads results/backtest.json (from forecast.backtest on the current snapshot), logs params and
metrics, fits the three quantile models on the whole snapshot, logs them as one pyfunc model that
maps a feature frame to q10 / q50 / q90, registers a new version, and moves the `champion` alias
to it only if its backtest MAE beats the current champion's on the same snapshot. A champion scored
on an older snapshot is re-scored on this one first. The decision goes to results/champion.json.

Run (stack up): uv run python -m forecast.train
"""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from forecast import backtest, features
from forecast.lgbm import LightGBMQuantile
from ingest.sources import school_holidays

EXPERIMENT = "paris-bike-forecast"
MODEL_NAME = "paris-bike-lgbm"
ALIAS = "champion"


def decide(
    candidate_mae: float,
    snapshot: str,
    champion: dict | None,
    rescore: Callable[[dict], float],
) -> dict:
    """Promotion rule. `champion` holds its version, mae, snapshot hash and params, or is None."""
    if champion is None:
        return {"promote": True, "reason": "no champion yet", "candidate_mae": candidate_mae}
    champion_mae = champion["mae"]
    rescored = champion["snapshot"] != snapshot
    if rescored:
        champion_mae = rescore(champion["params"])
    promote = candidate_mae < champion_mae
    return {
        "promote": promote,
        "reason": "beats the champion" if promote else "does not beat the champion",
        "candidate_mae": candidate_mae,
        "champion_version": champion["version"],
        "champion_mae": champion_mae,
        "champion_rescored": rescored,
    }


def rescore_params(params: dict) -> float:
    data = backtest.load_snapshot()
    cutoffs = backtest.fold_cutoffs(data["ds"].max(), backtest.FOLDS)
    result = backtest.run([LightGBMQuantile(params)], data, cutoffs, exog=backtest.load_exog())
    return result["overall"]["lightgbm"]["mae"]


def final_cutoff(data: pd.DataFrame) -> pd.Timestamp:
    """The hour after the last snapshot hour: the final model trains on everything before it."""
    return data["ds"].max() + pd.Timedelta(hours=1)


def _pyfunc_class():
    import mlflow.pyfunc

    class QuantileModel(mlflow.pyfunc.PythonModel):
        """Feature frame (forecast.features.FEATURES) -> q10, q50, q90."""

        def load_context(self, context):
            import lightgbm as lgb

            self.boosters = {
                name: lgb.Booster(model_file=path) for name, path in sorted(context.artifacts.items())
            }

        def predict(self, context, model_input, params=None):
            import numpy as np

            X = model_input[features.FEATURES].copy()
            X["counter"] = X["counter"].astype("category")
            preds = np.column_stack([self.boosters[n].predict(X) for n in ("q10", "q50", "q90")])
            preds = np.sort(np.clip(preds, 0, None), axis=1)
            return pd.DataFrame(preds, columns=["q10", "q50", "q90"], index=model_input.index)

    return QuantileModel


def main() -> None:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    result = json.loads((backtest.RESULTS / "backtest.json").read_text())
    snapshot = json.loads((backtest.RESULTS / "snapshot.json").read_text())
    if result["snapshot"]["counts_sha256"] != snapshot["counts_sha256"] or "lightgbm" not in result["overall"]:
        raise SystemExit("results/backtest.json is not a LightGBM run on the current snapshot; rerun forecast.backtest")
    snap_id = snapshot["counts_sha256"]
    overall = result["overall"]["lightgbm"]
    model = LightGBMQuantile()

    with mlflow.start_run(run_name="lightgbm-quantile") as run:
        mlflow.log_params({f"lgbm_{k}": v for k, v in model.params.items()})
        mlflow.log_params({"features": ",".join(features.FEATURES), "snapshot_sha256": snap_id,
                           "snapshot_end": snapshot["end"], "folds": len(result["folds"])})
        mlflow.log_metrics({f"backtest_{k}": v for k, v in overall.items() if k != "n"})
        for base in ("seasonal_naive", "seasonal_naive_4w"):
            if base in result["overall"]:
                mlflow.log_metric(f"{base}_mae", result["overall"][base]["mae"])
        for fold in result["folds"]:
            for k, v in fold["models"]["lightgbm"].items():
                if k != "n":
                    mlflow.log_metric(f"fold_{k}", v, step=fold["fold"])
        mlflow.log_artifact(str(backtest.RESULTS / "backtest.json"))

        data = backtest.load_snapshot()
        exog = backtest.load_exog()
        model.fit(data, final_cutoff(data), backtest.HORIZON, exog)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = {}
            for q, m in model.models.items():
                path = Path(tmp) / f"q{int(q * 100)}.txt"
                m.booster_.save_model(str(path))
                artifacts[path.stem] = str(path)
            example = features.build(
                data[data["ds"] < final_cutoff(data)], [final_cutoff(data).floor("D")], 24,
                exog["weather_fc"], exog["holidays"], model.counters,
                school_holidays.days(exog["school_holidays"]),
            )[features.FEATURES].head(5)
            example["counter"] = example["counter"].astype(str)  # the model casts it back
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=_pyfunc_class()(),
                artifacts=artifacts,
                code_paths=[str(Path(features.__file__).parent)],
                input_example=example,
                pip_requirements=["lightgbm==4.7.0", "pandas", "numpy"],
            )
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        for key, value in {"backtest_mae": overall["mae"], "snapshot_sha256": snap_id,
                           "params": json.dumps(model.params, sort_keys=True)}.items():
            client.set_model_version_tag(MODEL_NAME, version.version, key, str(value))

        try:
            current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            champion = {
                "version": current.version,
                "mae": float(current.tags["backtest_mae"]),
                "snapshot": current.tags["snapshot_sha256"],
                "params": json.loads(current.tags["params"]),
            }
        except MlflowException:  # no champion alias yet
            champion = None
        decision = decide(overall["mae"], snap_id, champion, rescore_params)
        if decision["promote"]:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        decision |= {"candidate_version": version.version, "snapshot_sha256": snap_id, "run_id": run.info.run_id}
        mlflow.log_dict(decision, "champion_decision.json")

    (backtest.RESULTS / "champion.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
