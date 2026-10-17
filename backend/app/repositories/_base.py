"""Shared PostgREST repository behavior and error translation."""

from __future__ import annotations

from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel
from supabase import AsyncClient

ModelT = TypeVar("ModelT", bound=BaseModel)


class RepositoryError(RuntimeError):
    """Raised when a Supabase persistence operation fails."""


class SupabaseRepository(Generic[ModelT]):
    """Small async PostgREST repository base with dependency injection."""

    table_name: str
    model_type: type[ModelT]

    def __init__(self, client: AsyncClient) -> None:
        self.client = client

    @staticmethod
    def _rows(response: Any) -> list[dict[str, Any]]:
        error = getattr(response, "error", None)
        if error:
            raise RepositoryError(str(error))
        data = getattr(response, "data", None)
        if data is None:
            return []
        if isinstance(data, dict):
            return [data]
        return list(data)

    async def _execute(self, operation: str, query: Any) -> list[dict[str, Any]]:
        try:
            response = await query.execute()
            return self._rows(response)
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(
                f"Supabase {operation} failed for table '{self.table_name}': {exc}"
            ) from exc

    @staticmethod
    def _uuid(value: UUID | str) -> str:
        try:
            return str(value if isinstance(value, UUID) else UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("record id must be a valid UUID") from exc

    def _model(self, row: dict[str, Any]) -> ModelT:
        try:
            return self.model_type.model_validate(row)
        except Exception as exc:
            raise RepositoryError(
                f"Supabase returned an invalid {self.table_name} record: {exc}"
            ) from exc
