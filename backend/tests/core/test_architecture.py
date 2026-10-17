import numpy as np
import pytest

from app.core.architecture import (
    SystemArchitectureRegistry,
    core_array_contract_telemetry,
    inspect_system_layout_telemetry,
    validate_core_array_contract,
)


@pytest.fixture(autouse=True)
def clean_registry():
    registry = SystemArchitectureRegistry.get_instance()
    registry.reset()
    yield registry
    registry.reset()


def test_empty_registry_is_singleton_and_reports_compliant_empty_layout(clean_registry):
    assert SystemArchitectureRegistry() is clean_registry
    assert clean_registry.get_registered_modules() == {}
    telemetry = inspect_system_layout_telemetry()
    assert telemetry["registry_empty"] is True
    assert telemetry["active_compute_modules"] == []
    assert telemetry["pure_engine_compliant"] is True
    assert telemetry["registered_module_count"] == 0
    assert telemetry["latency_ms"] >= 0.0


def test_registry_accepts_math_dependencies_and_tracks_execution_metadata(clean_registry):
    assert clean_registry.register_module(
        "app.core.portfolio.optimizer",
        dependencies=["numpy", "pandas", "scipy.optimize", "app.core.risk.metrics"],
        execution_metadata={"last_latency_ms": 1.2},
    )
    assert clean_registry.record_execution(
        "app.core.portfolio.optimizer", {"input_shape": [252, 4]}
    )
    validation = clean_registry.validate_isolation()
    assert validation["compliant"] is True
    module = clean_registry.get_registered_modules()["app.core.portfolio.optimizer"]
    assert module["package"] == "portfolio"
    assert module["execution_metadata"] == {
        "last_latency_ms": 1.2,
        "input_shape": [252, 4],
    }
    layout = inspect_system_layout_telemetry()
    assert layout["module_registration_status"]["app.core.portfolio.optimizer"] is True
    assert layout["package_module_counts"]["portfolio"] == 1


@pytest.mark.parametrize("dependency", ["fastapi", "supabase", "requests", "app.api.routes"])
def test_registry_detects_dependencies_outside_pure_math_boundary(clean_registry, dependency):
    clean_registry.register_module("risk.metrics", dependencies=["numpy", dependency])
    result = clean_registry.validate_isolation()
    assert result["compliant"] is False
    assert result["violations"][0]["dependency"] == dependency
    with pytest.raises(RuntimeError, match="isolation failed"):
        clean_registry.assert_isolation()


def test_registry_rejects_unknown_module_package_and_missing_execution_record():
    registry = SystemArchitectureRegistry()
    with pytest.raises(ValueError, match="module must belong"):
        registry.register_module("app.api.routes", dependencies=["numpy"])
    with pytest.raises(KeyError, match="not registered"):
        registry.record_execution("trading.unknown", {"latency_ms": 1.0})
    with pytest.raises(TypeError, match="not a string"):
        registry.register_module("risk.var", dependencies="numpy")


@pytest.mark.parametrize(
    ("array", "expected_dim", "allow_nan", "expected"),
    [
        (np.array([1.0, 2.0]), 1, False, True),
        (np.array([[1.0, 2.0]]), 1, False, False),
        (np.array([1.0, np.nan]), 1, False, False),
        (np.array([1.0, np.nan]), 1, True, True),
        (np.array([1.0, np.inf]), 1, True, False),
        (np.array([1, 2, 3], dtype=np.int64), 1, False, True),
        (np.array(["1", "2"]), 1, False, False),
        (np.array([1.0, 2.0, 3.0, 4.0])[::2], 1, False, False),
    ],
)
def test_array_contract_checks_dimensions_dtype_and_nonfinite_values(
    array, expected_dim, allow_nan, expected
):
    assert validate_core_array_contract(array, expected_dim, allow_nan) is expected


def test_array_contract_rejects_noncontiguous_and_unaligned_buffers():
    noncontiguous = np.arange(12.0).reshape(3, 4)[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    assert validate_core_array_contract(noncontiguous, 2) is False

    raw = bytearray(1 + 8 * 3)
    unaligned = np.ndarray((3,), dtype=np.float64, buffer=raw, offset=1)
    assert not unaligned.flags.aligned
    assert validate_core_array_contract(unaligned, 1) is False


def test_empty_and_invalid_expected_dimension_array_contracts():
    assert validate_core_array_contract(np.empty((0,), dtype=np.float64), 1) is False
    assert validate_core_array_contract(np.empty((0, 10), dtype=np.float64), 2) is False
    assert validate_core_array_contract(np.empty((10, 0), dtype=np.float64), 2) is False
    assert validate_core_array_contract(np.empty((2, 0, 3), dtype=np.float64), 3) is False
    assert validate_core_array_contract(np.array([1.0]), 1) is True
    assert validate_core_array_contract(np.arange(1_000_000, dtype=np.float64), 1) is True
    assert validate_core_array_contract(np.array(1.0), 0) is False
    assert validate_core_array_contract(np.array([1.0]), 2) is False
    assert validate_core_array_contract([1.0], 1) is False


def test_array_contract_telemetry_reports_validation_state_and_memory_layout():
    values = np.array([1.0, 2.0, 3.0])
    telemetry = core_array_contract_telemetry(values, expected_dim=1)
    assert telemetry["valid"] is True
    assert telemetry["validation_state"] == "valid"
    assert telemetry["input_shape"] == [3]
    assert telemetry["c_contiguous"] is True
    assert telemetry["aligned"] is True
    assert telemetry["nan_count"] == 0
    assert telemetry["latency_ms"] >= 0.0

    invalid = core_array_contract_telemetry(np.array([np.nan]), expected_dim=1)
    assert invalid["valid"] is False
    assert invalid["validation_state"] == "invalid"
    assert invalid["nan_count"] == 1
