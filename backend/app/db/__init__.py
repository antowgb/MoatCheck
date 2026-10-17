"""Database client interfaces for application persistence."""

from app.db.supabase import get_async_supabase_client

__all__ = ["get_async_supabase_client"]
