"""Integrity checks for bundled, source-documented historical crisis data."""

import numpy as np

from app.core.risk.crisis_scenarios import list_crisis_scenarios, load_crisis_scenario


def test_crisis_catalog_has_source_and_market_availability_notes() -> None:
    catalog = {item["id"]: item for item in list_crisis_scenarios()}
    assert set(catalog) == {"lehman-2008", "covid-2020"}
    assert "Yahoo Finance" in catalog["lehman-2008"]["source"]
    assert "BTC" not in catalog["lehman-2008"]["assets"]
    assert "URTH" in catalog["covid-2020"]["assets"]


def test_lehman_snapshot_contains_historical_market_losses_and_treasury_diversification() -> None:
    metadata, returns = load_crisis_scenario("lehman-2008")
    assert len(returns) == metadata["observations"] == 49
    assert returns.index[0].date().isoformat() == "2008-09-15"
    assert returns.index[-1].date().isoformat() == "2008-11-20"
    assert returns["SP500"].min() < -0.08
    assert (1.0 + returns["SP500"]).prod() - 1.0 < -0.35
    assert (1.0 + returns["IEF"]).prod() - 1.0 > 0.0
    assert np.isfinite(returns.to_numpy()).all()


def test_covid_snapshot_includes_mscI_world_proxy_and_bitcoin_returns() -> None:
    metadata, returns = load_crisis_scenario("covid-2020")
    assert len(returns) == metadata["observations"] == 51
    assert returns.index[0].date().isoformat() == "2020-02-19"
    assert returns.index[-1].date().isoformat() == "2020-04-30"
    assert set(returns.columns) == {"SP500", "URTH", "IEF", "BTC-USD"}
    assert returns["SP500"].min() < -0.1
    assert returns["BTC-USD"].min() < -0.25
    assert "not the licensed index" in metadata["asset_metadata"]["URTH"]["note"]


def test_unknown_scenario_is_rejected() -> None:
    try:
        load_crisis_scenario("not-a-scenario")
    except KeyError as error:
        assert "not-a-scenario" in str(error)
    else:
        raise AssertionError("unknown scenario ID should be rejected")
