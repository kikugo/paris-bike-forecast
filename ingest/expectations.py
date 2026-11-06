"""Great Expectations suites run on every load, before rows are written.

Hard rules fail the load (DataQualityError). Gaps are not checked here: they are expected in
this data and are measured by the gap report instead.
"""

from dataclasses import dataclass, field
from functools import cache

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd


class DataQualityError(RuntimeError):
    pass


@dataclass
class Validation:
    suite: str
    success: bool
    rows: int
    evaluated: int
    failed: int
    failures: list[str] = field(default_factory=list)

    def merge(self, other: "Validation") -> "Validation":
        return Validation(
            suite=self.suite,
            success=self.success and other.success,
            rows=self.rows + other.rows,
            evaluated=self.evaluated + other.evaluated,
            failed=self.failed + other.failed,
            failures=sorted(set(self.failures) | set(other.failures)),
        )


def _counts_suite() -> list:
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="counter_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="ts"),
        gxe.ExpectColumnValuesToNotBeNull(column="count"),
        gxe.ExpectColumnValuesToBeBetween(column="count", min_value=0),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["counter_id", "ts"]),
        # Seconds past the hour; every count is an hourly bucket.
        gxe.ExpectColumnValuesToBeInSet(column="ts_offset_s", value_set=[0]),
    ]


def _weather_suite() -> list:
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="ts"),
        gxe.ExpectColumnValuesToBeUnique(column="ts"),
        gxe.ExpectColumnValuesToBeInSet(column="ts_offset_s", value_set=[0]),
        gxe.ExpectColumnValuesToBeBetween(column="temperature_2m", min_value=-30, max_value=50),
        gxe.ExpectColumnValuesToBeBetween(column="relative_humidity_2m", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="precipitation", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="rain", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="wind_speed_10m", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="cloud_cover", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeInSet(column="source", value_set=["archive", "forecast"]),
    ]


def _weather_forecast_suite() -> list:
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="ts"),
        gxe.ExpectColumnValuesToNotBeNull(column="lead_day"),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["ts", "lead_day"]),
        gxe.ExpectColumnValuesToBeInSet(column="ts_offset_s", value_set=[0]),
        gxe.ExpectColumnValuesToBeBetween(column="lead_day", min_value=1, max_value=7),
        gxe.ExpectColumnValuesToBeBetween(column="temperature_2m", min_value=-30, max_value=50),
        gxe.ExpectColumnValuesToBeBetween(column="relative_humidity_2m", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="precipitation", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="wind_speed_10m", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="cloud_cover", min_value=0, max_value=100),
    ]


def _holidays_suite() -> list:
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="day"),
        gxe.ExpectColumnValuesToBeUnique(column="day"),
        gxe.ExpectColumnValuesToNotBeNull(column="name"),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
    ]


def _school_holidays_suite() -> list:
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="first_day"),
        gxe.ExpectColumnValuesToBeUnique(column="first_day"),
        gxe.ExpectColumnValuesToNotBeNull(column="last_day"),
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(column_A="last_day", column_B="first_day", or_equal=True),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
    ]


SUITES = {
    "counts_hourly": _counts_suite,
    "weather_hourly": _weather_suite,
    "weather_forecast": _weather_forecast_suite,
    "holidays": _holidays_suite,
    "school_holidays": _school_holidays_suite,
}


@cache
def _definition(suite_name: str):
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name=f"{suite_name}_source")
    asset = source.add_dataframe_asset(name=suite_name)
    batch = asset.add_batch_definition_whole_dataframe(name=f"{suite_name}_batch")
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name, expectations=SUITES[suite_name]()))
    return context.validation_definitions.add(
        gx.ValidationDefinition(name=f"{suite_name}_validation", data=batch, suite=suite)
    )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns:
        ts = out["ts"]
        out["ts_offset_s"] = ts.dt.minute * 60 + ts.dt.second
    return out


def validate(suite_name: str, df: pd.DataFrame) -> Validation:
    result = _definition(suite_name).run(batch_parameters={"dataframe": _prepare(df)})
    stats = result.statistics
    failures = [
        f"{r.expectation_config.type}({r.expectation_config.kwargs.get('column') or r.expectation_config.kwargs.get('column_list') or ''})"
        for r in result.results
        if not r.success
    ]
    return Validation(
        suite=suite_name,
        success=bool(result.success),
        rows=len(df),
        evaluated=int(stats["evaluated_expectations"]),
        failed=int(stats["unsuccessful_expectations"]),
        failures=failures,
    )


def require(validation: Validation) -> Validation:
    if not validation.success:
        raise DataQualityError(f"{validation.suite} failed: {', '.join(validation.failures)}")
    return validation
