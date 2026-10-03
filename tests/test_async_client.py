"""
Unit tests for asynchronous AsyncDeepSeekClient with mocked HTTP backend.
"""

import httpx
import pytest

from deepseek_web import AsyncDeepSeekClient, Session
from .conftest import create_mock_transport


@pytest.mark.asyncio
async def test_async_client_chat_and_stream():
    session = Session(token="mock_token")
    mock_http = httpx.AsyncClient(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )

    async with AsyncDeepSeekClient(session=session, http_client=mock_http) as client:
        # Non-streaming chat
        reply = await client.chat("Hi async")
        assert reply.text == "Hello there!"
        assert reply.message_id == 777
        assert reply.chat_session_id == "mock_session_100"

        # Streaming
        stream = client.stream("Hi again async")
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert stream.full_text == "Hello there!"
        assert stream.conversation_id == "mock_session_100:777"


@pytest.mark.asyncio
async def test_async_client_session_and_files():
    session = Session(token="mock_token")
    mock_http = httpx.AsyncClient(
        base_url="https://chat.deepseek.com",
        transport=create_mock_transport(),
    )

    async with AsyncDeepSeekClient(session=session, http_client=mock_http) as client:
        # Profile
        profile = await client.get_user_profile()
        assert profile["name"] == "Test User"

        # Session create and delete
        sid = await client.create_chat_session()
        assert sid == "mock_session_100"

        sessions = await client.list_chat_sessions()
        assert len(sessions) == 1

        deleted = await client.delete_chat_session(sid)
        assert deleted is True

        # File upload
        res = await client.upload_file(b"async content", filename="test.txt")
        assert res.file_id == "file_uploaded_99"

        prepared_id = await client.upload_and_prepare_file(b"async image", filename="img.png", for_vision=True)
        assert prepared_id == "file_forked_99"
