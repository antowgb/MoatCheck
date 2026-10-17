"""Cross-module regression contracts discovered by the platform audit."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.api.v1.endpoints._utils import json_safe
from app.core.risk.crisis_scenarios import list_crisis_scenarios, load_crisis_scenario
from app.core.risk.metrics import portfolio_var_cvar


@pytest.mark.parametrize("method", ["historical", "parametric"])
def test_positive_loss_contract_floors_profitable_tails_at_zero(method):
    result = portfolio_var_cvar(pd.DataFrame({"A": [.01] * 10}), [1.0], method=method)
    assert result["var"] == result["cvar"] == 0.0


def test_transport_serializes_undefined_ratios_without_nonstandard_json():
    result = json_safe({"metrics": np.array([np.inf, np.nan, -np.inf, .25]), "scalar": np.float64(np.inf)})
    assert result == {"metrics": [None, None, None, .25], "scalar": None}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scenario", ["lehman-2008", "covid-2020"])
def test_embedded_history_matches_canonical_artifacts_and_checksums(scenario):
    directory = Path(__file__).parents[2] / "app/core/risk/data"
    manifest = json.loads((directory / "manifest.json").read_text())
    artifact = directory / f"{scenario}.csv"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact_sha256"][artifact.name]
    expected = pd.read_csv(artifact, index_col="date", parse_dates=["date"])
    _, actual = load_crisis_scenario(scenario)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-15)


def test_static_crisis_data_has_no_shared_mutable_result_state():
    first = list_crisis_scenarios()
    first[0]["limitations"].clear()
    first[0]["assets"].clear()
    assert list_crisis_scenarios()[0]["limitations"]
    metadata, panel = load_crisis_scenario("lehman-2008")
    panel.iloc[0, 0] = 999
    assert load_crisis_scenario("lehman-2008")[1].iloc[0, 0] != 999
    assert metadata["assets"]


def test_core_has_no_application_service_imports_or_direct_data_io():
    """Static boundary guard, with libraries and native extension imports permitted."""
    root = Path(__file__).parents[2] / "app/core"
    forbidden_imports = {"fastapi", "supabase", "httpx", "requests", "urllib", "socket", "psycopg", "sqlalchemy"}
    forbidden_calls = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "read_csv", "read_json", "read_sql", "to_sql", "to_csv"}
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if module.split(".")[0] in forbidden_imports or module.startswith(("app.services", "app.db", "app.repositories", "app.api")):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {module}")
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in forbidden_calls:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    assert violations == []
