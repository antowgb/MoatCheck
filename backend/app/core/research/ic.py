"""Pure cross-sectional information-coefficient calculations.

Factor scores are indexed by asset and forward returns by the same asset
identifiers. For panels, rows represent signal dates and columns represent
assets. Missing pairs are omitted independently at each date.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _aligned_pair(
    factor_scores: pd.Series, forward_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    r"""Validate, align by asset label, and return observed numeric pairs.

    For the shared label set :math:`A`, retain pairs
    :math:`(x_a,y_a)` only when both values are present. Pairwise alignment
    estimates :math:`\operatorname{Corr}(x_a,y_a)` for the same asset and
    avoids matching one stock's score with another's return. For four stocks,
    each label's score is paired with that same label's next-period return.
    Duplicate labels are rejected because they make matching ambiguous.
    Complexity: :math:`O(n\log n)` time for index alignment and :math:`O(n)`
    space.
    """
    if not isinstance(factor_scores, pd.Series) or not isinstance(forward_returns, pd.Series):
        raise TypeError("factor_scores and forward_returns must be pandas Series")
    for name, series in (("factor_scores", factor_scores), ("forward_returns", forward_returns)):
        if not series.index.is_unique:
            raise ValueError(f"{name} index must be unique")
        try:
            values = series.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        observed = values[~np.isnan(values)]
        if not np.isfinite(observed).all():
            raise ValueError(f"{name} must be finite wherever present")

    aligned = pd.concat(
        [factor_scores.rename("factor"), forward_returns.rename("forward_return")],
        axis=1,
        join="inner",
    ).dropna()
    values = aligned.to_numpy(dtype=float, copy=True)
    if values.shape[0] < 2:
        raise ValueError("at least two aligned, non-missing pairs are required")
    return values[:, 0], values[:, 1]


def _correlation(x: np.ndarray, y: np.ndarray, *, rank: bool) -> float:
    r"""Calculate one correlation, returning NaN for a constant cross-section.

    Pearson uses :math:`\operatorname{Cov}(x,y)/(s_xs_y)`; Spearman applies
    this correlation to ranked values. For four stocks, matching rankings
    yield :math:`+1` and reversed rankings yield :math:`-1`. A constant input
    has no defined correlation, so it is represented by NaN.
    Complexity is :math:`O(n\log n)` for Spearman ranking and :math:`O(n)` for
    Pearson; both require :math:`O(n)` temporary space.
    """
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    result = spearmanr(x, y) if rank else pearsonr(x, y)
    return float(result.statistic)


def calculate_rank_ic(factor_scores: pd.Series, forward_returns: pd.Series) -> float:
    r"""Calculate Spearman Rank IC between factor scores and forward returns.

    The statistic is :math:`IC_{rank}=\operatorname{Corr}(\operatorname{rank}(x_t),
    \operatorname{rank}(R_{t+1}))`. It measures whether the factor orders
    assets in the same order as their subsequent returns, without requiring a
    linear relationship. For four stocks with factor ranks ``[1, 2, 3, 4]``
    and forward-return ranks ``[1, 2, 3, 4]``, Rank IC is ``+1``; if the return
    ranks are ``[4, 3, 2, 1]``, it is ``-1``. Complexity:
    :math:`O(n\log n)` time and :math:`O(n)` space for ``n`` assets.
    """
    x, y = _aligned_pair(factor_scores, forward_returns)
    return _correlation(x, y, rank=True)


def calculate_normal_ic(factor_scores: pd.Series, forward_returns: pd.Series) -> float:
    r"""Calculate Pearson IC between factor scores and forward returns.

    The statistic is :math:`IC_{normal}=\frac{\operatorname{Cov}(x,R)}
    {\sigma_x\sigma_R}`. Unlike Rank IC, it measures linear association and
    can be influenced by extreme observations. In the four-stock example,
    scores ``[1, 2, 3, 4]`` and returns ``[1, 2, 3, 4]`` yield ``+1``; reversing
    the returns yields ``-1``. Complexity: :math:`O(n)` time and
    :math:`O(n)` space.
    """
    x, y = _aligned_pair(factor_scores, forward_returns)
    return _correlation(x, y, rank=False)


def _aligned_panels(
    factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Validate and align factor and forward-return panels on date and asset.

    Shared cells represent valid score/forward-return candidates; NaNs are
    handled pairwise within each date. For four stocks at one date, the four
    scores are paired with those same four stocks' forward returns. Unique row
    and column labels are required to preserve one-to-one alignment. Complexity:
    :math:`O(TN\log(TN))` worst-case alignment time and :math:`O(TN)` space
    for ``T`` dates and ``N`` assets.
    """
    if not isinstance(factor_df, pd.DataFrame) or not isinstance(forward_returns_df, pd.DataFrame):
        raise TypeError("factor_df and forward_returns_df must be pandas DataFrames")
    for name, frame in (("factor_df", factor_df), ("forward_returns_df", forward_returns_df)):
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"{name} must have unique date and asset labels")
        try:
            values = frame.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values") from exc
        observed = values[~np.isnan(values)]
        if not np.isfinite(observed).all():
            raise ValueError(f"{name} must be finite wherever present")
    factors, returns = factor_df.align(forward_returns_df, join="inner", axis=None)
    return factors.astype(float), returns.astype(float)


def calculate_ic_time_series(
    factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame
) -> pd.Series:
    r"""Calculate cross-sectional Rank IC separately for each signal date.

    At date :math:`t`, the result is
    :math:`IC_t=\operatorname{Corr}_{a}(\operatorname{rank}(x_{a,t}),
    \operatorname{rank}(R_{a,t+1}))`, using the shared assets with non-missing
    pairs. This produces a history of factor predictability rather than one
    pooled statistic. For four stocks, perfectly matching score/return ranks
    on a date give ``+1``, while a complete reversal gives ``-1``. Dates with
    fewer than two valid assets or a constant series produce NaN. Complexity:
    :math:`O(TN\log N)` time and :math:`O(TN)` space.
    """
    factors, returns = _aligned_panels(factor_df, forward_returns_df)
    values: list[float] = []
    for date in factors.index:
        pair = pd.concat([factors.loc[date], returns.loc[date]], axis=1).dropna()
        if len(pair) < 2:
            values.append(float("nan"))
        else:
            x = pair.iloc[:, 0].to_numpy(dtype=float)
            y = pair.iloc[:, 1].to_numpy(dtype=float)
            values.append(_correlation(x, y, rank=True))
    return pd.Series(values, index=factors.index, name="rank_ic", dtype=float)


def calculate_information_ratio(ic_series: pd.Series) -> dict[str, float | int]:
    r"""Summarize an IC history and calculate its information ratio.

    For observed period ICs :math:`IC_t`, report
    :math:`\overline{IC}=T^{-1}\sum_t IC_t`, sample standard deviation
    :math:`s_{IC}`, and :math:`IR=\overline{IC}/s_{IC}`. IR scales average
    predictive strength by its variability. A factor with positive IC in all
    four stocks at each date has a positive mean; a consistently reversed
    ranking has a negative mean. If the IC history is empty, an error is
    raised; if its standard deviation is zero, IR is NaN because the ratio is
    undefined. Complexity: :math:`O(T)` time and :math:`O(T)` space.
    """
    if not isinstance(ic_series, pd.Series):
        raise TypeError("ic_series must be a pandas Series")
    try:
        values = ic_series.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError("ic_series must contain numeric values") from exc
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("IC values must be finite wherever present")
    if observed.size == 0:
        raise ValueError("ic_series must contain at least one non-missing value")
    mean_ic = float(np.mean(observed))
    std_ic = float(np.std(observed, ddof=1)) if observed.size >= 2 else float("nan")
    information_ratio = mean_ic / std_ic if std_ic > 0 else float("nan")
    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "information_ratio": float(information_ratio),
        "observations": int(observed.size),
    }


def calculate_ic_time_series_with_telemetry(
    factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame
) -> tuple[pd.Series, dict[str, float | int]]:
    r"""Calculate Rank IC history and diagnostics for execution and data coverage.

    Telemetry reports elapsed milliseconds :math:`1000(t_1-t_0)`, evaluated
    date count :math:`T`, valid paired cells, and coverage
    :math:`C=N_{valid}/N_{possible}` over the aligned panel. For example, if
    four assets are present on two dates and six score/return pairs are
    non-missing, coverage is :math:`6/8=0.75`. This makes missingness visible
    alongside IC values. Complexity: :math:`O(TN\log N)` time and
    :math:`O(TN)` space.
    """
    start = perf_counter()
    factors, returns = _aligned_panels(factor_df, forward_returns_df)
    ic_series = calculate_ic_time_series(factors, returns)
    valid_pairs = int((factors.notna() & returns.notna()).to_numpy().sum())
    possible_pairs = int(factors.size)
    return ic_series, {
        "latency_ms": (perf_counter() - start) * 1000.0,
        "time_periods_evaluated": int(len(ic_series)),
        "valid_pairs": valid_pairs,
        "possible_pairs": possible_pairs,
        "coverage_ratio": valid_pairs / possible_pairs if possible_pairs else 0.0,
    }


def information_ratio_with_telemetry(
    ic_series: pd.Series,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    r"""Calculate IC summary and telemetry for its number of valid periods.

    Coverage is the fraction of non-missing IC observations in the supplied
    time series, :math:`C=N_{valid}/N_{periods}`; latency is reported in
    milliseconds. These diagnostics distinguish a strong estimate with broad
    data coverage from one based on only a few dates. Complexity:
    :math:`O(T)` time and :math:`O(T)` space.
    """
    start = perf_counter()
    summary = calculate_information_ratio(ic_series)
    elapsed_ms = (perf_counter() - start) * 1000.0
    total_periods = int(len(ic_series))
    return summary, {
        "latency_ms": elapsed_ms,
        "time_periods_evaluated": total_periods,
        "valid_pairs": int(summary["observations"]),
        "possible_pairs": total_periods,
        "coverage_ratio": summary["observations"] / total_periods if total_periods else 0.0,
    }
