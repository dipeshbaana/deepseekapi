"""
Unit tests for data models and exceptions.
"""

import os
import stat
import tempfile
import time
from pathlib import Path

from deepseek_web.models import (
    APIResponseError,
    AuthenticationError,
    AVAILABLE_MODELS,
    ChatReply,
    ChatStreamChunk,
    DEFAULT_MODEL,
    DeepSeekError,
    FileUploadResult,
    PoWChallengeError,
    RateLimitError,
    Session,
)


def test_session_lifecycle_and_permissions():
    """Test Session creation, serialization, age, expiry, and secure 0600 file permissions."""
    session = Session(
        token="test-secret-token-123",
        cookies={"ds_session": "abc456"},
        user_agent="CustomUA/1.0",
        captured_at=time.time(),
    )

    assert session.token == "test-secret-token-123"
    assert session.cookies["ds_session"] == "abc456"
    assert session.user_agent == "CustomUA/1.0"
    assert session.age >= 0.0
    assert not session.is_expired(max_age_seconds=3600)

    # Test saving to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        sess_path = Path(tmpdir) / "test_session.json"
        session.save(sess_path)

        assert sess_path.exists()

        # Check POSIX permissions (should be 0600 on Linux/POSIX)
        if os.name == "posix":
            file_mode = stat.S_IMODE(sess_path.stat().st_mode)
            assert file_mode == 0o600, f"Expected 0600, got {oct(file_mode)}"

        # Load session back
        loaded = Session.load(sess_path)
        assert loaded is not None
        assert loaded.token == session.token
        assert loaded.cookies == session.cookies
        assert loaded.user_agent == session.user_agent


def test_session_expiration():
    """Test expired session detection."""
    old_session = Session(
        token="old-token",
        captured_at=time.time() - 4000,
    )
    assert old_session.is_expired(max_age_seconds=3600)


def test_chat_reply_and_chunks():
    """Test ChatReply and ChatStreamChunk properties."""
    reply = ChatReply(
        text="Hello world!",
        reasoning_content="Thinking process...",
        conversation_id="sess_123:45",
        chat_session_id="sess_123",
        message_id=45,
    )
    assert str(reply) == "Hello world!"
    assert reply.conversation_id == "sess_123:45"
    assert reply.reasoning_content == "Thinking process..."

    chunk1 = ChatStreamChunk(text="delta text", is_thinking=False)
    assert chunk1.has_content
    assert not chunk1.is_thinking

    chunk2 = ChatStreamChunk(reasoning_content="delta reasoning", is_thinking=True)
    assert chunk2.has_content
    assert chunk2.is_thinking

    chunk3 = ChatStreamChunk()
    assert not chunk3.has_content


def test_available_models():
    """Verify built-in models configuration."""
    assert DEFAULT_MODEL in AVAILABLE_MODELS
    assert "deepseek-chat" in AVAILABLE_MODELS
    assert "deepseek-reasoner" in AVAILABLE_MODELS

    chat_opt = AVAILABLE_MODELS["deepseek-chat"]
    assert chat_opt.model_type == "default"
    assert not chat_opt.thinking_enabled

    reasoner_opt = AVAILABLE_MODELS["deepseek-reasoner"]
    assert reasoner_opt.model_type == "default"
    assert reasoner_opt.thinking_enabled


def test_exception_hierarchy():
    """Verify DeepSeek exception classes."""
    assert issubclass(AuthenticationError, DeepSeekError)
    assert issubclass(PoWChallengeError, DeepSeekError)
    assert issubclass(APIResponseError, DeepSeekError)
    assert issubclass(RateLimitError, DeepSeekError)

    err = APIResponseError(1001, "Invalid parameter", raw_data={"detail": "foo"})
    assert "1001" in str(err)
    assert err.code == 1001
    assert err.raw_data == {"detail": "foo"}
