"""WebSocket handler for real-time status, telemetry, and events.

Provides a single /ws endpoint with channel-based subscriptions:
- "hardware": periodic hardware stats (CPU%, RAM, GPU)
- "jobs": job state change notifications
- "status": backend status updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.services.hardware_manager import HardwareManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

VALID_CHANNELS = {"hardware", "jobs", "status"}


class ConnectionManager:
    """Manages WebSocket connections and channel subscriptions."""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, set[str]] = {}

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> None:
        """Accept a WebSocket connection."""
        await ws.accept()
        self._connections[ws] = set()
        logger.info("WebSocket connected (total: %d)", self.connection_count)

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self._connections.pop(ws, None)
        logger.info("WebSocket disconnected (total: %d)", self.connection_count)

    def subscribe(self, ws: WebSocket, channel: str) -> bool:
        """Subscribe a connection to a channel.

        Returns:
            True if valid channel, False otherwise.
        """
        if channel not in VALID_CHANNELS:
            return False
        if ws in self._connections:
            self._connections[ws].add(channel)
            logger.debug("WS subscribed to '%s'", channel)
        return True

    def unsubscribe(self, ws: WebSocket, channel: str) -> bool:
        """Unsubscribe a connection from a channel."""
        if channel not in VALID_CHANNELS:
            return False
        if ws in self._connections:
            self._connections[ws].discard(channel)
        return True

    def get_subscriptions(self, ws: WebSocket) -> set[str]:
        """Get channels a connection is subscribed to."""
        return self._connections.get(ws, set())

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send data to all connections subscribed to a channel."""
        message = json.dumps({"channel": channel, "data": data})
        disconnected: list[WebSocket] = []

        for ws, channels in self._connections.items():
            if channel in channels:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(message)
                except Exception:
                    disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, data: dict[str, Any]) -> None:
        """Send a message to a specific connection."""
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(json.dumps(data))
        except Exception as exc:
            logger.debug("Failed to send to WS: %s", exc)


# Singleton connection manager
ws_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time telemetry and events.

    Protocol:
      Client sends JSON: {"action": "subscribe"|"unsubscribe", "channel": "hardware"|"jobs"|"status"}
      Server sends JSON: {"channel": "...", "data": {...}}
    """
    await ws_manager.connect(websocket)
    hw_task: asyncio.Task[None] | None = None

    try:
        # Send welcome message
        await ws_manager.send_to(websocket, {
            "type": "connected",
            "channels": list(VALID_CHANNELS),
            "message": "Connected to ReelMotion AI WebSocket",
        })

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_to(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            action = msg.get("action", "")
            channel = msg.get("channel", "")

            if action == "subscribe":
                if ws_manager.subscribe(websocket, channel):
                    await ws_manager.send_to(websocket, {
                        "type": "subscribed",
                        "channel": channel,
                    })
                    # Start hardware telemetry task if subscribing to hardware
                    if channel == "hardware" and hw_task is None:
                        hw_manager: HardwareManager = websocket.app.state.hardware_manager
                        interval = websocket.app.state.settings.hardware.monitoring_interval_seconds
                        hw_task = asyncio.create_task(
                            _hardware_telemetry_loop(websocket, hw_manager, interval)
                        )
                else:
                    await ws_manager.send_to(websocket, {
                        "type": "error",
                        "message": f"Invalid channel: '{channel}'. Valid: {list(VALID_CHANNELS)}",
                    })

            elif action == "unsubscribe":
                if ws_manager.unsubscribe(websocket, channel):
                    await ws_manager.send_to(websocket, {
                        "type": "unsubscribed",
                        "channel": channel,
                    })
                    # Cancel hardware task if unsubscribing
                    if channel == "hardware" and hw_task is not None:
                        hw_task.cancel()
                        hw_task = None
                else:
                    await ws_manager.send_to(websocket, {
                        "type": "error",
                        "message": f"Invalid channel: '{channel}'",
                    })

            elif action == "ping":
                await ws_manager.send_to(websocket, {
                    "type": "pong",
                    "timestamp": time.time(),
                })

            else:
                await ws_manager.send_to(websocket, {
                    "type": "error",
                    "message": f"Unknown action: '{action}'",
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        if hw_task is not None:
            hw_task.cancel()
        ws_manager.disconnect(websocket)


async def _hardware_telemetry_loop(
    ws: WebSocket,
    hw_manager: HardwareManager,
    interval: int,
) -> None:
    """Send periodic hardware stats to a subscribed WebSocket."""
    try:
        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break
            if "hardware" not in ws_manager.get_subscriptions(ws):
                break

            stats = hw_manager.get_live_stats()
            await ws_manager.send_to(ws, {
                "channel": "hardware",
                "data": stats,
            })
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("Hardware telemetry loop ended: %s", exc)
