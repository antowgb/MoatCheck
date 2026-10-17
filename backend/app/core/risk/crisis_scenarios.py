"""Versioned, package-local historical benchmark return snapshots.

The module builds fresh arrays from versioned static Python constants and makes
no filesystem, HTTP, database, or runtime market-data requests. Canonical
CSV/JSON provenance artifacts remain in ``data/``; tests verify their checksums
and exact numerical parity with the embedded constants.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from app.core.risk._crisis_data import MANIFEST, PANELS




def list_crisis_scenarios() -> list[dict[str, Any]]:
    """Return JSON-safe scenario/source metadata without loading return files."""
    manifest = deepcopy(MANIFEST)
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "start_date": item["start"],
            "end_date": item["end"],
            "base_close_date": item["base_date"],
            "observations": item["observations"],
            "assets": item["assets"],
            "asset_metadata": {ticker: manifest["assets"][ticker] for ticker in item["assets"]},
            "source": manifest["data_source"],
            "retrieved_on": manifest["retrieved_on"],
            "calendar_note": manifest["calendar"],
            "limitations": manifest["limitations"],
        }
        for item in manifest["scenarios"]
    ]


def load_crisis_scenario(scenario_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load one checked-in daily return panel and its provenance metadata.

    The returned DataFrame has a strictly increasing date index and one
    finite simple-return column per actually available benchmark. For example,
    the 2008 panel contains the S&P 500, a VT global-equity proxy and IEF; BTC
    is absent because it was not a traded asset then. Complexity is
    :math:`O(TA)` time and memory for ``T`` observations and ``A`` assets.
    """
    item = next((entry for entry in MANIFEST["scenarios"] if entry["id"] == scenario_id), None)
    if item is None:
        raise KeyError(f"unknown historical crisis scenario: {scenario_id}")
    panel = PANELS[scenario_id]
    frame = pd.DataFrame(panel["returns"], columns=item["assets"], index=pd.to_datetime(panel["dates"]))
    frame.index.name = "date"
    if frame.empty or list(frame.columns) != item["assets"]:
        raise ValueError(f"invalid static historical data artifact for {scenario_id}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"historical dates must be unique and increasing for {scenario_id}")
    values = frame.to_numpy(dtype=float, copy=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"historical returns must be a finite matrix for {scenario_id}")
    metadata = next(meta for meta in list_crisis_scenarios() if meta["id"] == scenario_id)
    return metadata, frame
