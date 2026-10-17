"""Pure date guards for point-in-time and survivorship-safe backtests.

These functions only filter and validate in-memory Pandas data. Dates are
compared in UTC; timezone-naive values are interpreted as UTC. Missing
publication or listing dates cannot establish eligibility and are excluded.
"""

from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd


def _as_utc_timestamp(value: object, name: str) -> pd.Timestamp:
    r"""Parse one valid timestamp and normalize it to UTC.

    :math:`t_{UTC}=t` for naive dates and :math:`t_{UTC}=t\,\text{converted}
    \text{ to UTC}` for timezone-aware dates. Consistent timezone handling
    prevents invalid date comparisons; a Q4 period ending Dec 31 still remains
    unavailable until its Feb 15 publication timestamp. Complexity:
    :math:`O(1)` time and space.
    """
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _date_column(df: pd.DataFrame, column: str) -> pd.Series:
    r"""Parse a required date column as UTC while preserving missing entries.

    A malformed non-missing date is rejected rather than silently treated as
    missing; NaT remains NaT so the caller can apply a conservative eligibility
    rule. For example, a Q4 `period_end` of Dec 31 and `know_date` of Feb 15
    remain correctly ordered through this conversion. Complexity:
    :math:`O(n)` time and :math:`O(n)` space.
    """
    if column not in df.columns:
        raise ValueError(f"required date column '{column}' is missing")
    try:
        parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"column '{column}' must contain valid timestamps") from exc
    invalid = df[column].notna() & parsed.isna()
    if bool(invalid.any()):
        raise ValueError(f"column '{column}' contains invalid timestamps")
    return parsed


def filter_point_in_time_data(
    df: pd.DataFrame,
    current_backtest_date: pd.Timestamp,
    know_date_col: str = "know_date",
) -> pd.DataFrame:
    r"""Keep only records available to an investor on the backtest date.

    Eligibility is :math:`\text{know\_date}_i\le t_{backtest}`. The
    publication/knowledge date, not the fiscal period end, controls when a
    fundamental value can enter a historical portfolio. For example, Q4 ends
    Dec 31 but is reported Feb 15: it is excluded for a Jan 31 backtest and
    included on Feb 15. Rows with missing ``know_date`` are conservatively
    excluded. Complexity: :math:`O(n)` time and :math:`O(n)` mask space.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    backtest_date = _as_utc_timestamp(current_backtest_date, "current_backtest_date")
    know_dates = _date_column(df, know_date_col)
    return df.loc[know_dates.notna() & (know_dates <= backtest_date)].copy()


def apply_survivorship_filter(
    universe_df: pd.DataFrame,
    current_backtest_date: pd.Timestamp,
    listing_date_col: str = "first_listed",
    delisting_date_col: str = "delisted_date",
) -> pd.DataFrame:
    r"""Select assets that existed in the investable universe at date ``t``.

    An asset is eligible when :math:`\text{first\_listed}\le t` and
    :math:`(\text{delisted\_date is NaT})\lor(\text{delisted\_date}\ge t)`.
    This includes historical companies on the dates they were actually listed
    and avoids rebuilding the past universe from companies that survive today.
    A missing delisting date means still active; a missing listing date is
    excluded. For instance, an active stock with a Q4 report disclosed Feb 15
    can appear in a Feb 15 universe while a company delisted Feb 14 cannot.
    Complexity: :math:`O(n)` time and :math:`O(n)` mask space.
    """
    if not isinstance(universe_df, pd.DataFrame):
        raise TypeError("universe_df must be a pandas DataFrame")
    backtest_date = _as_utc_timestamp(current_backtest_date, "current_backtest_date")
    listing_dates = _date_column(universe_df, listing_date_col)
    delisting_dates = _date_column(universe_df, delisting_date_col)
    listed = listing_dates.notna() & (listing_dates <= backtest_date)
    not_yet_delisted = delisting_dates.isna() | (delisting_dates >= backtest_date)
    return universe_df.loc[listed & not_yet_delisted].copy()


def _point_in_time_violations(
    factor_df: pd.DataFrame,
    know_date_col: str,
    period_end_col: str,
) -> tuple[int, int]:
    r"""Count complete observations whose publication precedes period end.

    A valid row satisfies :math:`\text{know\_date}\ge\text{period\_end}`.
    The Dec 31/Feb 15 Q4 example is valid because the filing came after the
    period ended; reversing those dates is lookahead leakage. Incomplete date
    pairs raise because integrity cannot be established. Complexity:
    :math:`O(n)` time and :math:`O(n)` temporary space.
    """
    if not isinstance(factor_df, pd.DataFrame):
        raise TypeError("factor_df must be a pandas DataFrame")
    know_dates = _date_column(factor_df, know_date_col)
    period_ends = _date_column(factor_df, period_end_col)
    missing = know_dates.isna() | period_ends.isna()
    if bool(missing.any()):
        raise ValueError("know_date and period_end must be present for every record")
    violations = int(np.count_nonzero((know_dates < period_ends).to_numpy()))
    return violations, int(len(factor_df))


def validate_point_in_time_integrity(
    factor_df: pd.DataFrame,
    know_date_col: str = "know_date",
    period_end_col: str = "period_end",
) -> bool:
    r"""Validate that no factor is recorded as known before its period ended.

    Integrity requires :math:`\text{know\_date}\ge\text{period\_end}` for
    every row. A Q4 record with period end Dec 31 and release date Feb 15
    passes; a Dec 20 knowledge date for that Dec 31 period is leakage and
    raises :class:`ValueError`. Equality is valid because the requirement is
    inclusive. Empty input passes vacuously. Complexity: :math:`O(n)` time and
    :math:`O(n)` temporary space.
    """
    violations, _ = _point_in_time_violations(factor_df, know_date_col, period_end_col)
    if violations:
        raise ValueError(f"detected {violations} lookahead violation(s): know_date precedes period_end")
    return True


def filter_point_in_time_data_with_telemetry(
    df: pd.DataFrame,
    current_backtest_date: pd.Timestamp,
    know_date_col: str = "know_date",
) -> tuple[pd.DataFrame, dict[str, float | int | list[int]]]:
    r"""Filter point-in-time rows and report latency and row-count diagnostics.

    Telemetry includes :math:`1000(t_1-t_0)` milliseconds, input/output row
    counts and shapes, and the number of excluded rows. The Dec 31 Q4 value
    reported Feb 15 appears in the output only when ``current_backtest_date``
    reaches Feb 15. Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    start = perf_counter()
    result = filter_point_in_time_data(df, current_backtest_date, know_date_col)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return result, {
        "latency_ms": elapsed_ms,
        "input_rows": int(len(df)),
        "filtered_rows": int(len(result)),
        "excluded_rows": int(len(df) - len(result)),
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "output_shape": [int(result.shape[0]), int(result.shape[1])],
    }


def apply_survivorship_filter_with_telemetry(
    universe_df: pd.DataFrame,
    current_backtest_date: pd.Timestamp,
    listing_date_col: str = "first_listed",
    delisting_date_col: str = "delisted_date",
) -> tuple[pd.DataFrame, dict[str, float | int | list[int]]]:
    r"""Apply the historical-universe filter and expose its row-count impact.

    Returns processing latency :math:`1000(t_1-t_0)`, input and retained asset
    counts, excluded asset count, and matrix shapes. A company delisted on
    Feb 14 is absent on Feb 15, while an active Q4 reporter with NaT delisting
    remains eligible if it was already listed. Complexity: :math:`O(n)` time
    and :math:`O(n)` space.
    """
    start = perf_counter()
    result = apply_survivorship_filter(
        universe_df,
        current_backtest_date,
        listing_date_col=listing_date_col,
        delisting_date_col=delisting_date_col,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    return result, {
        "latency_ms": elapsed_ms,
        "input_rows": int(len(universe_df)),
        "filtered_rows": int(len(result)),
        "excluded_rows": int(len(universe_df) - len(result)),
        "input_shape": [int(universe_df.shape[0]), int(universe_df.shape[1])],
        "output_shape": [int(result.shape[0]), int(result.shape[1])],
    }


def validate_point_in_time_integrity_with_telemetry(
    factor_df: pd.DataFrame,
    know_date_col: str = "know_date",
    period_end_col: str = "period_end",
) -> tuple[bool, dict[str, float | int | list[int]]]:
    r"""Validate date integrity and report any lookahead violations found.

    Returns ``False`` instead of raising for detected leakage and records the
    count :math:`N_{violations}=|\{i:know_i<period_i\}|`, processing latency,
    rows checked, and input shape. A Feb 15 filing for a Dec 31 Q4 period is
    valid and contributes zero violations. Malformed or missing dates still
    raise because the check cannot classify them safely. Complexity:
    :math:`O(n)` time and :math:`O(n)` space.
    """
    start = perf_counter()
    violations, rows_checked = _point_in_time_violations(
        factor_df, know_date_col, period_end_col
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    return violations == 0, {
        "latency_ms": elapsed_ms,
        "rows_checked": rows_checked,
        "detected_lookahead_violations": violations,
        "input_shape": [int(factor_df.shape[0]), int(factor_df.shape[1])],
    }
