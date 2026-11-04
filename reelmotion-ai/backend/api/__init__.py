"""API layer: FastAPI routers for health, hardware, characters, recordings, rendering, models, vision."""

from backend.api import characters, health, models, recording, rendering, vision, websocket

__all__ = [
    "characters",
    "health",
    "models",
    "recording",
    "rendering",
    "vision",
    "websocket",
]
