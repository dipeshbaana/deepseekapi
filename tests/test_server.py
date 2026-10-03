"""
Unit tests for OpenAI-compatible FastAPI server endpoints.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from deepseek_web import AsyncDeepSeekClient, Session
from deepseek_web.server.app import app, set_client
from .conftest import create_mock_transport


@pytest.fixture(autouse=True)
def setup_mock_client():
    """Inject a mocked AsyncDeepSeekClient into the server app."""
    session = Session(token="mock_token_for_server")
    mock_http = httpx.AsyncClient(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )
    client = AsyncDeepSeekClient(session=session, http_client=mock_http)
    set_client(client)
    yield
    set_client(None)


def test_healthz_endpoint():
    with TestClient(app) as test_client:
        resp = test_client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["authenticated"] is True


def test_list_models_endpoint():
    with TestClient(app) as test_client:
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "deepseek-chat" in model_ids
        assert "deepseek-reasoner" in model_ids


def test_chat_completions_non_streaming():
    with TestClient(app) as test_client:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Hello server"}
            ],
            "stream": False,
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello there!"
        assert data["conversation_id"] == "mock_session_100:777"


def test_chat_completions_streaming():
    with TestClient(app) as test_client:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "Stream to me"}
            ],
            "stream": True,
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n\n")
        assert len(lines) >= 2
        # Final line is [DONE]
        assert "data: [DONE]" in resp.text


def test_upload_file_endpoint():
    with TestClient(app) as test_client:
        files = {"file": ("test.txt", b"file content", "text/plain")}
        resp = test_client.post("/v1/files", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "file"
        assert data["id"] == "file_uploaded_99"
        assert data["filename"] == "test.txt"


def test_api_key_protection(monkeypatch):
    monkeypatch.setenv("SERVER_API_KEY", "super-secret-123")
    with TestClient(app) as test_client:
        # Request without header fails 401
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

        # Request with correct header succeeds
        resp2 = test_client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer super-secret-123"},
        )
        assert resp2.status_code == 200
