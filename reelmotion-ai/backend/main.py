"""ReelMotion AI — FastAPI Application Entry Point.

Local-first AI Character Replacement & Reels Recording Studio.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.api import (
    characters,
    exports,
    health,
    models,
    recording,
    rendering,
    websocket,
)
from backend.core.config import get_settings
from backend.core.logging import (
    RequestLoggingMiddleware,
    get_logger,
    setup_logging,
)
from backend.services.hardware_manager import HardwareManager
from backend.services.job_manager import JobManager
from backend.services.model_manager import ModelManager

logger = get_logger("reelmotion.main")

# Background task for periodic telemetry broadcast
_telemetry_task: asyncio.Task[None] | None = None


async def _telemetry_broadcast_loop(hw_manager: HardwareManager, interval: float = 5.0) -> None:
    """Periodically broadcast hardware stats over WebSocket to subscribed clients."""
    while True:
        try:
            await asyncio.sleep(interval)
            if websocket.ws_manager.connection_count > 0:
                stats = hw_manager.get_live_stats()
                await websocket.ws_manager.broadcast("hardware", stats)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Error in telemetry broadcast loop: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    global _telemetry_task

    # 1. Record startup time
    start_time = time.time()
    health.set_start_time(start_time)

    # 2. Load configuration
    settings = get_settings()
    app.state.settings = settings

    # 3. Setup logging
    setup_logging(level=settings.server.log_level, json_format=False)
    logger.info("Starting ReelMotion AI v%s", __version__)

    # 4. Initialize hardware manager & perform real detection
    hw_manager = HardwareManager(force_cpu=settings.hardware.force_cpu)
    profile = hw_manager.detect()
    app.state.hardware_manager = hw_manager
    logger.info(
        "Hardware detected: %s (VRAM: %d MB), Tier: %s, CPU: %s (%d cores), RAM: %d MB",
        profile.gpu.name,
        profile.gpu.vram_total_mb,
        profile.tier.value,
        profile.cpu.name,
        profile.cpu.logical_cores,
        profile.ram.total_mb,
    )

    # 5. Initialize model manager
    model_manager = ModelManager()
    app.state.model_manager = model_manager
    logger.info("Loaded model manifest with %d declared models", len(model_manager.list_models()))

    # 6. Initialize job manager
    job_manager = JobManager()
    app.state.job_manager = job_manager

    # 7. Ensure data directories exist
    resolved = settings.resolved_paths()
    for name, path in resolved.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Verified directory %s: %s", name, path)

    # 8. Start telemetry background broadcast task
    _telemetry_task = asyncio.create_task(_telemetry_broadcast_loop(hw_manager))

    logger.info("ReelMotion AI server initialized and ready.")
    yield

    # Shutdown
    logger.info("Shutting down ReelMotion AI server...")
    if _telemetry_task:
        _telemetry_task.cancel()
        try:
            await _telemetry_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ReelMotion AI",
        description="Local-first AI Character Replacement & Reels Recording Studio",
        version=__version__,
        lifespan=lifespan,
    )

    # Attach settings early so tests can access them before startup lifespan runs
    app.state.settings = settings
    app.state.hardware_manager = HardwareManager(force_cpu=settings.hardware.force_cpu)
    app.state.model_manager = ModelManager()
    app.state.job_manager = JobManager()

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Logging Middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Include Routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(characters.router)
    app.include_router(recording.router)
    app.include_router(rendering.router)
    app.include_router(exports.router)
    app.include_router(websocket.router)

    @app.get("/", tags=["root"])
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "name": "ReelMotion AI",
                "version": __version__,
                "status": "online",
                "docs_url": "/docs",
            }
        )

    return app


app = create_app()


def run() -> None:
    """CLI runner."""
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.server.log_level.lower(),
    )


if __name__ == "__main__":
    run()

