"""Versioned HTTP API, isolated from the pure quantitative core."""

from fastapi import APIRouter

from app.api.v1.endpoints.backtest import router as backtest_router
from app.api.v1.endpoints.advisor import router as advisor_router
from app.api.v1.endpoints.portfolio import router as portfolio_router, portfolio_collection_router
from app.api.v1.endpoints.risk import router as risk_router
from app.api.v1.endpoints.session import router as session_router
from app.api.v1.endpoints.privacy import router as privacy_router
from app.api.v1.endpoints.screener import router as screener_router
from app.api.v1.endpoints.universe import router as universe_router

router = APIRouter()
router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
router.include_router(portfolio_collection_router, prefix="/portfolios", tags=["portfolio"])
router.include_router(risk_router, prefix="/risk", tags=["risk"])
router.include_router(backtest_router, prefix="/backtest", tags=["backtest"])
router.include_router(advisor_router, prefix="/advisor", tags=["advisor"])
router.include_router(session_router, tags=["session"])
router.include_router(privacy_router, tags=["privacy"])
router.include_router(screener_router, tags=["screener"])
router.include_router(universe_router, tags=["market-universe"])

__all__ = ["router"]
