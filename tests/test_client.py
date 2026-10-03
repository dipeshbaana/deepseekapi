"""
Unit tests for synchronous DeepSeekClient with mocked HTTP backend.
"""

import httpx
import pytest

from deepseek_web import DeepSeekClient, Session
from .conftest import create_mock_transport


def test_client_chat_and_stream():
    session = Session(token="mock_token")
    mock_http = httpx.Client(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )

    with DeepSeekClient(session=session, http_client=mock_http) as client:
        # Non-streaming chat
        reply = client.chat("Hi")
        assert reply.text == "Hello there!"
        assert reply.message_id == 777
        assert reply.chat_session_id == "mock_session_100"
        assert reply.conversation_id == "mock_session_100:777"

        # Streaming
        stream = client.stream("Hi again", conversation_id=reply.conversation_id)
        chunks = list(stream)
        assert len(chunks) == 2
        assert stream.full_text == "Hello there!"


def test_client_session_management():
    session = Session(token="mock_token")
    mock_http = httpx.Client(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )

    with DeepSeekClient(session=session, http_client=mock_http) as client:
        # User profile
        profile = client.get_user_profile()
        assert profile["email"] == "user@example.com"

        # Create session
        sess_id = client.create_chat_session()
        assert sess_id == "mock_session_100"

        # List sessions
        sessions = client.list_chat_sessions(count=5)
        assert len(sessions) == 1
        assert sessions[0].id == "mock_session_100"

        # Delete session
        deleted = client.delete_chat_session("mock_session_100")
        assert deleted is True


def test_client_file_upload_and_prepare():
    session = Session(token="mock_token")
    mock_http = httpx.Client(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )

    with DeepSeekClient(session=session, http_client=mock_http) as client:
        # Direct upload
        res = client.upload_file(b"test document content", filename="doc.txt")
        assert res.file_id == "file_uploaded_99"

        # Upload & prepare
        fid = client.upload_and_prepare_file(b"sample image", filename="img.png", for_vision=True)
        assert fid == "file_forked_99"
