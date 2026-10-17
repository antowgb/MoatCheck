"""FastAPI entry point.

Also serves the statically exported Next.js frontend (``static/`` directory,
produced by ``npm run build`` on the frontend side) — a single service in prod.
"""

import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.v2 import router as v2_router
from app.api.v1 import router as v1_router

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

production = os.getenv("APP_ENV", "development").lower() == "production"
app = FastAPI(
    title="Quant Stock Screener",
    version="0.1.0",
    docs_url=None if production else "/docs",
    redoc_url=None if production else "/redoc",
    openapi_url=None if production else "/openapi.json",
)


@app.exception_handler(ValueError)
@app.exception_handler(TypeError)
@app.exception_handler(np.linalg.LinAlgError)
async def invalid_quant_input(request: Request, exc: Exception) -> JSONResponse:
    """Map core array and mathematical contract failures to HTTP 422."""
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "invalid_quant_input", "message": "Invalid input or calculation parameters."}},
    )


@app.exception_handler(Exception)
async def log_unhandled_exceptions(request: Request, exc: Exception) -> JSONResponse:
    """Logs the full traceback for any unhandled exception before returning 500.

    Without this, an unhandled exception in a route just becomes Starlette's
    generic "Internal Server Error" with nothing in the logs to diagnose it —
    this keeps that same response but makes the cause debuggable.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

configured_origins = os.getenv("FRONTEND_ORIGINS")
if configured_origins is None:
    origins = [] if production else ["http://localhost:3000"]
else:
    origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

from app.security.middleware import (  # noqa: E402
    APIRequestProtectionMiddleware,
    CookieCSRFMiddleware,
    SecurityHeadersMiddleware,
    csrf_settings,
    middleware_settings,
)

security_settings = middleware_settings()
static_dir = Path(__file__).resolve().parent.parent / "static"
app.add_middleware(APIRequestProtectionMiddleware, **security_settings)
app.add_middleware(CookieCSRFMiddleware, **csrf_settings())
app.add_middleware(SecurityHeadersMiddleware, static_dir=static_dir, production=production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "HEAD", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Key", "X-MoatCheck-Request"],
    allow_credentials=True,
)

app.include_router(router, prefix="/api")
app.include_router(v2_router, prefix="/api/v2")
app.include_router(v1_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Static frontend (Next.js export copied into backend/static at build time).
# Mounted last so it doesn't shadow /api and /health.
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
else:
    logging.getLogger(__name__).info(
        "No static/ directory — frontend not served (API-only dev mode)."
    )
