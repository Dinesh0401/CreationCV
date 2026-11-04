"""Tests for WebSocket real-time communication."""

import json
import pytest
from fastapi.testclient import TestClient


def test_websocket_connect_and_welcome(client: TestClient):
    """Test connecting to /ws and receiving initial welcome message."""
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg.get("type") == "connected"
        assert "channels" in msg
        assert "hardware" in msg["channels"]


def test_websocket_subscription(client: TestClient):
    """Test subscribing to a valid channel."""
    with client.websocket_connect("/ws") as ws:
        # Welcome message
        ws.receive_json()

        # Subscribe to hardware channel
        ws.send_json({"action": "subscribe", "channel": "hardware"})
        res = ws.receive_json()
        assert res.get("type") == "subscribed"
        assert res.get("channel") == "hardware"


def test_websocket_invalid_channel_failure(client: TestClient):
    """Failure test: Subscribing to an unknown channel returns error."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"action": "subscribe", "channel": "invalid_fake_channel"})
        res = ws.receive_json()
        assert res.get("type") == "error"
        assert "invalid channel" in res.get("message", "").lower()


def test_websocket_invalid_json_failure(client: TestClient):
    """Failure test: Sending malformed non-JSON text returns JSON error."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_text("this is not json { [")
        res = ws.receive_json()
        assert res.get("type") == "error"
        assert "invalid json" in res.get("message", "").lower()
