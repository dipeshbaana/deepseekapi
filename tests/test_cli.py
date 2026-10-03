"""
Unit tests for CLI commands and argument parsing.
"""

import sys
import pytest

from deepseek_web.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        sys.argv = ["deepseek-web", "--help"]
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "CLI tool for DeepSeek Web Client" in captured.out


def test_cli_sessions_list(monkeypatch, capsys):
    from deepseek_web.models import ChatSessionItem
    import deepseek_web.cli as cli

    class MockClient:
        def __init__(self, **kwargs):
            pass

        def list_chat_sessions(self, count=20):
            return [
                ChatSessionItem(id="sess_123", title="Test Session", created_at=0, updated_at=0)
            ]

    monkeypatch.setattr(cli, "DeepSeekClient", MockClient)

    sys.argv = ["deepseek-web", "sessions", "list"]
    main()
    captured = capsys.readouterr()
    assert "sess_123" in captured.out
    assert "Test Session" in captured.out
