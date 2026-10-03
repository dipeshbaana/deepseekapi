"""
Unit tests for authentication module.
"""

import os
import tempfile
from pathlib import Path
import pytest

from deepseek_web.auth import get_session
from deepseek_web.models import AuthenticationError, Session


def test_auth_explicit_token():
    session = get_session(token="explicit_token_123")
    assert session.token == "explicit_token_123"


def test_auth_env_token(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TOKEN", "env_token_456")
    monkeypatch.setenv("DEEPSEEK_COOKIES", '{"test_cookie": "val"}')

    session = get_session()
    assert session.token == "env_token_456"
    assert session.cookies.get("test_cookie") == "val"


def test_auth_from_saved_file(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    monkeypatch.delenv("DEEPSEEK_USER_TOKEN", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        sess_file = Path(tmpdir) / "session.json"
        saved = Session(token="saved_token_789")
        saved.save(sess_file)

        session = get_session(session_file=sess_file)
        assert session.token == "saved_token_789"


def test_auth_non_interactive_missing_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    monkeypatch.delenv("DEEPSEEK_USER_TOKEN", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_file = Path(tmpdir) / "non_existent.json"
        non_existent_profile = Path(tmpdir) / "non_existent_prof"

        with pytest.raises(AuthenticationError):
            get_session(
                session_file=non_existent_file,
                profile_dir=non_existent_profile,
                allow_interactive=False,
            )
