import numpy as np
import pandas as pd
import pytest

from app.core.profiling.benchmarks import (
    BenchmarkInput,
    ComplexityAnalyzer,
    benchmark_pipeline_phases,
    profile_execution_time,
)


def test_profile_decorator_preserves_output_and_records_shapes_and_latency():
    expected = {"score": 0.125}

    @profile_execution_time
    def compute(values, *, frame):
        assert values.shape == (4,)
        assert frame.shape == (3, 2)
        return expected

    result = compute(np.arange(4.0), frame=pd.DataFrame(np.ones((3, 2))))
    assert result is expected
    assert compute.__name__ == "compute"
    telemetry = compute.last_telemetry
    assert telemetry["success"] is True
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["peak_latency_ms"] >= telemetry["latency_ms"]
    assert telemetry["input_shapes"]["args"][0]["shape"] == [4]
    assert telemetry["input_shapes"]["kwargs"]["frame"]["shape"] == [3, 2]
    assert telemetry["output_shape"]["kind"] == "dict"
    assert len(compute.telemetry_history) == 1


def test_profile_decorator_records_failed_call_and_reraises_exception():
    @profile_execution_time
    def fail(values):
        raise ArithmeticError("intentional failure")

    with pytest.raises(ArithmeticError, match="intentional"):
        fail(np.array([1.0]))
    assert fail.last_telemetry["success"] is False
    assert fail.last_telemetry["exception_type"] == "ArithmeticError"
    assert fail.last_telemetry["input_shapes"]["args"][0]["shape"] == [1]


def test_profile_decorator_captures_input_shape_before_in_place_mutation():
    @profile_execution_time
    def shrink(values):
        values.resize(1, refcheck=False)
        return values

    values = np.arange(4.0)
    result = shrink(values)
    assert result.shape == (1,)
    assert shrink.last_telemetry["input_shapes"]["args"][0]["shape"] == [4]
    assert shrink.last_telemetry["output_shape"]["shape"] == [1]


def test_complexity_analyzer_estimates_quadratic_growth():
    def quadratic_work(values):
        total = 0
        for left in range(len(values)):
            for right in range(len(values)):
                total += left * right
        return total

    analyzer = ComplexityAnalyzer(
        quadratic_work,
        input_factory=lambda n: np.linspace(0.0, 1.0, n),
        repeats=3,
        warmups=1,
    )
    result = analyzer.analyze([1600, 200, 800, 400])
    assert result["status"] == "estimated"
    assert result["gamma"] == pytest.approx(2.0, abs=0.35)
    assert result["empirical_exponent"] == result["gamma"]
    assert 0.8 <= result["r_squared"] <= 1.0
    assert result["input_sizes"] == [200, 400, 800, 1600]
    assert result["peak_latency_ms"] == max(
        row["max_latency_ms"] for row in result["measurements"]
    )
    assert result["measurements"][-1]["input_shapes"]["args"][0]["shape"] == [1600]
    assert result["throughput_per_second"] > 0.0


def test_single_size_and_single_element_input_return_gracefully_without_gamma():
    analyzer = ComplexityAnalyzer(lambda values: values.sum(), repeats=1, warmups=0)
    result = analyzer.analyze([1])
    assert result["status"] == "insufficient_sizes"
    assert result["gamma"] is None
    assert result["measurements"][0]["input_shapes"]["args"][0]["shape"] == [1]
    assert result["throughput_per_second"] > 0.0


def test_empty_or_invalid_size_inputs_raise_clear_errors():
    analyzer = ComplexityAnalyzer(lambda values: values.sum())
    with pytest.raises(ValueError, match="at least one"):
        analyzer.analyze([])
    with pytest.raises(ValueError, match="positive integer"):
        analyzer.analyze([0, 10])
    with pytest.raises(ValueError, match="unique"):
        analyzer.analyze([10, 10])
    with pytest.raises(ValueError, match="non-empty list"):
        benchmark_pipeline_phases(lambda x: x, lambda x: x, lambda x: x, [])


def test_custom_input_factory_can_supply_args_and_kwargs():
    analyzer = ComplexityAnalyzer(
        lambda values, scale=1.0: np.sum(values) * scale,
        input_factory=lambda n: BenchmarkInput(
            (np.ones(n, dtype=float),), {"scale": 2.0}
        ),
        repeats=1,
        warmups=0,
    )
    result = analyzer.analyze([2, 4])
    assert result["measurements"][0]["output_shape"]["shape"] == []
    assert result["measurements"][1]["input_shapes"]["kwargs"]["scale"]["shape"] == []

    tuple_contract = ComplexityAnalyzer(
        lambda values: values.shape[0],
        input_factory=lambda n: (np.ones(n), {}),
        repeats=1,
        warmups=0,
    ).analyze([3])
    assert tuple_contract["measurements"][0]["output_shape"]["shape"] == []


def test_pipeline_benchmark_returns_phase_exponents_throughput_and_shapes():
    result = benchmark_pipeline_phases(
        factor_func=lambda series: series.mean(),
        opt_func=lambda covariance: np.linalg.inv(covariance.to_numpy()),
        risk_func=lambda returns: returns.std(),
        sizes=[2, 4, 8],
    )
    assert set(result["phases"]) == {"factor", "optimization", "risk"}
    assert set(result["throughput_per_second"]) == {"factor", "optimization", "risk"}
    assert result["peak_latency_ms"] > 0.0
    assert result["total_benchmark_latency_ms"] >= result["peak_latency_ms"]
    assert result["input_array_shapes"]["factor"][0]["args"][0]["shape"] == [2]
    assert result["input_array_shapes"]["optimization"][0]["args"][0]["shape"] == [2, 2]
    assert result["input_array_shapes"]["risk"][0]["args"][0]["shape"] == [2]

