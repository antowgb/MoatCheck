"""In-memory architecture registry and array-boundary contracts for core math.

This module deliberately performs no filesystem or network inspection. It
validates declared module dependencies and numeric array contracts at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CORE_PACKAGES = frozenset({"factors", "portfolio", "risk", "trading"})
_PURE_DEPENDENCY_ROOTS = frozenset(
    {
        "abc", "array", "bisect", "builtins", "collections", "contextlib",
        "copy", "dataclasses", "datetime", "decimal", "enum", "fractions",
        "functools", "itertools", "math", "numbers", "operator", "random",
        "statistics", "time", "typing", "weakref", "warnings", "threading",
        "numpy", "pandas", "scipy", "statsmodels", "sklearn", "hmmlearn",
    }
)


@dataclass(frozen=True)
class ModuleRegistration:
    r"""Record declared dependencies and runtime metadata for one compute module.

    The record models a pure compute node :math:`y=f(x)` plus declared edges
    to other nodes or approved math libraries. For example,
    ``app.core.risk.metrics`` may declare ``numpy`` and ``scipy`` without
    coupling to an API or database layer. Constructing or copying a record is
    :math:`O(D+M)` in dependency count :math:`D` and metadata size :math:`M`.
    """

    module_name: str
    package: str
    dependencies: tuple[str, ...] = ()
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)


def _module_package(module_name: str) -> str | None:
    r"""Extract a recognized core package from a module path of length :math:`L`.

    ``app.core.portfolio.optimizer`` maps to ``portfolio``; a simple package
    path such as ``risk.garch`` maps to ``risk``. This supports a small
    numerical example while preventing arbitrary application layers from
    being registered as pure compute modules. Complexity: :math:`O(L)` time
    to split and inspect the path, with :math:`O(L)` temporary space.
    """
    parts = module_name.split(".")
    if len(parts) >= 3 and parts[:2] == ["app", "core"]:
        return parts[2]
    return parts[0] if parts else None


def _dependency_violation(dependency: str) -> str | None:
    r"""Return a policy violation for one declared dependency, if any, in O(1).

    Core modules may use approved numerical libraries and other registered
    core packages. Dependencies such as :math:`\texttt{fastapi}` or
    :math:`\texttt{supabase}` cross the pure-math boundary and are rejected.
    """
    normalized = dependency.strip().lower()
    if not normalized:
        return "dependency name cannot be empty"
    if normalized == "app.core":
        return None
    if normalized.startswith("app.core."):
        package = normalized.split(".")[2]
        if package in CORE_PACKAGES:
            return None
        return f"unapproved internal dependency: {dependency}"
    if normalized.startswith("app."):
        return f"application-layer dependency crosses core boundary: {dependency}"
    root = normalized.split(".", 1)[0]
    if root in _PURE_DEPENDENCY_ROOTS:
        return None
    return f"unapproved or I/O-capable dependency: {dependency}"


class SystemArchitectureRegistry:
    r"""Singleton registry enforcing declared pure-core module boundaries.

    The intended execution graph is a pure pipeline :math:`x\to f_1(x)
    \to f_2(f_1(x))`, with no network, persistence, filesystem, or web edges.
    For example, a portfolio optimizer can register NumPy/SciPy dependencies;
    registering FastAPI or Supabase marks the graph noncompliant. Registry
    operations are :math:`O(D)` for :math:`D` declared dependencies and use
    :math:`O(M+D)` memory for :math:`M` modules.
    """

    _instance: SystemArchitectureRegistry | None = None
    _instance_lock = RLock()

    def __new__(cls) -> SystemArchitectureRegistry:
        r"""Return the one process-local registry instance in :math:`O(1)`.

        Reusing the same registry prevents separate callers from seeing
        conflicting dependency graphs. For example, two constructors both
        return the same object. Complexity: :math:`O(1)` time and space.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        r"""Initialize the in-memory module map once in :math:`O(1)`.

        No external resources are opened. An empty registry is valid and
        reports zero active compute modules. Complexity: :math:`O(1)` time
        and space.
        """
        if getattr(self, "_initialized", False):
            return
        self._records: dict[str, ModuleRegistration] = {}
        self._lock = RLock()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> SystemArchitectureRegistry:
        r"""Return the singleton registry; complexity is :math:`O(1)`.

        This named accessor is convenient for module-level telemetry helpers.
        A typical numerical pipeline gets the same registry no matter which
        package requests it. Complexity: :math:`O(1)` time and space.
        """
        return cls()

    def reset(self) -> None:
        r"""Clear module declarations and execution metadata in :math:`O(M)`.

        Reset is intended for isolated test runs or explicit engine lifecycle
        management; it performs no I/O. Complexity: :math:`O(M)` time and
        :math:`O(1)` additional space for :math:`M` registered modules.
        """
        with self._lock:
            self._records.clear()

    def register_module(
        self,
        module_name: str,
        dependencies: Iterable[str] = (),
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        r"""Register or replace one core module declaration in memory.

        The module must belong to ``factors``, ``portfolio``, ``risk``, or
        ``trading``. Declaring dependencies does not import them; it records
        edges for later isolation checks. For example,
        ``register_module('portfolio.optimizer', ['numpy', 'scipy'])`` returns
        ``True``. Complexity: :math:`O(D+M)` to copy :math:`D` dependencies
        and metadata of size :math:`M`; lookup/insertion is expected :math:`O(1)`.
        """
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("module_name must be a non-empty string")
        name = module_name.strip()
        package = _module_package(name)
        if package not in CORE_PACKAGES:
            raise ValueError(
                f"module must belong to one of {sorted(CORE_PACKAGES)}"
            )
        if isinstance(dependencies, str):
            raise TypeError("dependencies must be an iterable of module names, not a string")
        try:
            deps = tuple(str(dependency).strip() for dependency in dependencies)
        except TypeError as exc:
            raise TypeError("dependencies must be an iterable of module names") from exc
        metadata = dict(execution_metadata or {})
        record = ModuleRegistration(name, package, deps, metadata)
        with self._lock:
            self._records[name] = record
        return True

    def record_execution(
        self, module_name: str, execution_metadata: Mapping[str, Any]
    ) -> bool:
        r"""Merge measured runtime metadata into a registered module in :math:`O(M)`.

        A caller can record values such as ``latency_ms`` or input dimensions
        after a calculation. For instance, a factor transform may attach its
        sample count and elapsed milliseconds. Complexity: :math:`O(M)` to
        copy metadata of size :math:`M` and :math:`O(1)` expected lookup.
        """
        with self._lock:
            if module_name not in self._records:
                raise KeyError(f"module is not registered: {module_name}")
            previous = self._records[module_name]
            metadata = dict(previous.execution_metadata)
            metadata.update(dict(execution_metadata))
            self._records[module_name] = ModuleRegistration(
                previous.module_name,
                previous.package,
                previous.dependencies,
                metadata,
            )
        return True

    def get_registered_modules(self) -> dict[str, dict[str, Any]]:
        r"""Return detached module metadata keyed by module path in :math:`O(M+D)`.

        Copies keep callers from mutating the active registry accidentally.
        An empty registry returns ``{}``; after registering a math module, its
        dependency list and runtime fields are visible. Complexity:
        :math:`O(M+D)` time and output space.
        """
        with self._lock:
            return {
                name: {
                    "module_name": record.module_name,
                    "package": record.package,
                    "dependencies": list(record.dependencies),
                    "execution_metadata": dict(record.execution_metadata),
                    "registered": True,
                }
                for name, record in self._records.items()
            }

    def validate_isolation(self) -> dict[str, Any]:
        r"""Check that every registered dependency respects the pure-core boundary.

        Compliance means all declared graph edges stay within recognized core
        packages or approved mathematical/standard libraries. An example edge
        from ``risk.var`` to ``supabase`` is reported as a violation. Runtime is
        :math:`O(E)` for :math:`E` declared dependency edges; output space is
        :math:`O(V+E)` for modules :math:`V` and violations.
        """
        started = perf_counter()
        violations: list[dict[str, str]] = []
        with self._lock:
            records = tuple(self._records.values())
        for record in records:
            for dependency in record.dependencies:
                message = _dependency_violation(dependency)
                if message is not None:
                    violations.append(
                        {
                            "module": record.module_name,
                            "dependency": dependency,
                            "reason": message,
                        }
                    )
        return {
            "compliant": not violations,
            "registered_module_count": len(records),
            "violations": violations,
            "latency_ms": float((perf_counter() - started) * 1000.0),
        }

    def assert_isolation(self) -> bool:
        r"""Raise if any registered dependency breaks isolation, in :math:`O(E)`.

        This method is suitable for a startup or deployment gate while
        remaining independent of the web application. A clean math-only
        registry returns ``True``; a dependency on a database raises
        ``RuntimeError``. Complexity: :math:`O(E)` time and :math:`O(V)` space.
        """
        result = self.validate_isolation()
        if not result["compliant"]:
            raise RuntimeError(f"core architecture isolation failed: {result['violations']}")
        return True


def validate_core_array_contract(
    arr: np.ndarray,
    expected_dim: int,
    allow_nan: bool = False,
) -> bool:
    r"""Validate a real numeric, aligned, C-contiguous array at a core boundary.

    A valid input obeys :math:`\operatorname{ndim}(A)=d`, has a supported
    real numeric dtype, aligned memory, C-contiguous layout, and finite values
    except optional NaNs. For example, ``np.array([1.0, 2.0])`` satisfies a
    one-dimensional contract, while ``np.array([[1.0, 2.0]])`` does not; NaN
    is rejected unless ``allow_nan=True``. This prevents malformed buffers
    from reaching vectorized kernels. Complexity: :math:`O(N)` time to inspect
    :math:`N` elements and :math:`O(1)` additional space.
    """
    if not isinstance(arr, np.ndarray):
        return False
    if not isinstance(allow_nan, (bool, np.bool_)):
        return False
    if isinstance(expected_dim, bool) or not isinstance(expected_dim, int) or expected_dim < 1:
        return False
    if arr.ndim != expected_dim:
        return False
    if any(size == 0 for size in arr.shape):
        return False
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        return False
    if not arr.flags.c_contiguous or not arr.flags.aligned:
        return False
    if not np.isfinite(arr).all():
        if not allow_nan or not np.issubdtype(arr.dtype, np.floating):
            return False
        if np.isinf(arr).any():
            return False
    return True


def core_array_contract_telemetry(
    arr: np.ndarray,
    expected_dim: int,
    allow_nan: bool = False,
) -> dict[str, Any]:
    r"""Return array validation state, layout, shape, and validation latency.

    For example, profiling a 1,000-element contiguous float vector reports its
    shape, dtype, alignment flags, NaN count, and whether it meets the core
    contract. Complexity: :math:`O(N)` time and :math:`O(1)` additional space.
    """
    started = perf_counter()
    is_array = isinstance(arr, np.ndarray)
    nan_count = 0
    if is_array and np.issubdtype(arr.dtype, np.floating):
        nan_count = int(np.isnan(arr).sum())
    valid = validate_core_array_contract(arr, expected_dim, allow_nan=allow_nan)
    return {
        "valid": bool(valid),
        "validation_state": "valid" if valid else "invalid",
        "expected_dim": expected_dim,
        "input_shape": list(arr.shape) if is_array else None,
        "dtype": str(arr.dtype) if is_array else None,
        "c_contiguous": bool(arr.flags.c_contiguous) if is_array else False,
        "aligned": bool(arr.flags.aligned) if is_array else False,
        "nan_count": nan_count,
        "allow_nan": bool(allow_nan),
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def inspect_system_layout_telemetry() -> dict[str, Any]:
    r"""Summarize the declared architecture registry and array memory standards.

    This function intentionally inspects only in-memory declarations rather
    than scanning files or importing application modules. Its pure pipeline
    model is :math:`x\mapsto f_n(\cdots f_2(f_1(x)))`; no I/O edges are
    permitted. For example, an empty registry reports zero active modules and
    remains compliant, while any registered forbidden dependency changes the
    compliance status to false. Complexity: :math:`O(E+V)` time and output
    space for registered dependency edges :math:`E` and modules :math:`V`.
    """
    started = perf_counter()
    registry = SystemArchitectureRegistry.get_instance()
    validation = registry.validate_isolation()
    modules = registry.get_registered_modules()
    package_counts = {package: 0 for package in sorted(CORE_PACKAGES)}
    for record in modules.values():
        package_counts[record["package"]] += 1
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "active_compute_modules": sorted(modules),
        "registered_module_count": len(modules),
        "module_registration_status": {
            name: bool(record["registered"]) for name, record in modules.items()
        },
        "package_module_counts": package_counts,
        "memory_alignment_standards": {
            "required_c_contiguous": True,
            "required_aligned": True,
            "accepted_dtype": "real integer or floating point",
            "infinity_allowed": False,
            "nan_allowed_by_default": False,
        },
        "numerical_stack_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "pure_engine_compliant": bool(validation["compliant"]),
        "isolation_violations": validation["violations"],
        "registry_empty": not modules,
    }


__all__ = [
    "CORE_PACKAGES",
    "ModuleRegistration",
    "SystemArchitectureRegistry",
    "core_array_contract_telemetry",
    "inspect_system_layout_telemetry",
    "validate_core_array_contract",
]
