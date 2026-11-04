"""Tests for recording endpoints."""

import io
import pytest
from httpx import ASGITransport, AsyncClient
from backend.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_recordings_lifecycle(app, tmp_path):
    app.state.settings.paths.recordings_dir = str(tmp_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. List initially empty
        resp = await client.get("/api/recordings")
        assert resp.status_code == 200
        assert resp.json() == []

        # 2. Upload recording
        dummy_file = io.BytesIO(b"dummy video data content")
        files = {"file": ("test_take.mp4", dummy_file, "video/mp4")}
        data = {
            "title": "Test Reel Take",
            "duration": "12.5",
            "fps": "30",
            "resolution": "720x1280",
        }
        upload_resp = await client.post("/api/recordings/upload", files=files, data=data)
        assert upload_resp.status_code == 201
        rec_data = upload_resp.json()
        assert rec_data["name"] == "Test Reel Take"
        assert rec_data["duration_seconds"] == 12.5
        rec_id = rec_data["id"]

        # 3. Retrieve recording by ID
        get_resp = await client.get(f"/api/recordings/{rec_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == rec_id

        # 4. Stream file
        file_resp = await client.get(f"/api/recordings/{rec_id}/file")
        assert file_resp.status_code == 200
        assert file_resp.content == b"dummy video data content"

        # 5. Delete recording
        del_resp = await client.delete(f"/api/recordings/{rec_id}")
        assert del_resp.status_code == 200

        # 6. Verify deleted
        get_after_del = await client.get(f"/api/recordings/{rec_id}")
        assert get_after_del.status_code == 404
