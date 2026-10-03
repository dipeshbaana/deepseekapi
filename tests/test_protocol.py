"""
Unit tests for protocol helpers, headers, and SSE streaming parser.
"""

import pytest

from deepseek_web.models import APIResponseError, Session
from deepseek_web.protocol import (
    build_base_headers,
    decode_conversation_id,
    encode_conversation_id,
    parse_sse_line,
    resolve_model_parameters,
    unwrap_biz_data,
    SSEParserState,
)


def test_build_base_headers():
    session = Session(token="my-token-xyz", user_agent="TestAgent/1.0")
    headers = build_base_headers(session)

    assert headers["authorization"] == "Bearer my-token-xyz"
    assert headers["user-agent"] == "TestAgent/1.0"
    assert headers["x-client-platform"] == "web"
    assert "origin" in headers
    assert "referer" in headers


def test_unwrap_biz_data():
    # Successful response with biz_data
    success_resp = {
        "code": 0,
        "msg": None,
        "data": {
            "biz_data": {"id": "session_123", "status": "ok"}
        }
    }
    biz = unwrap_biz_data(success_resp)
    assert biz["id"] == "session_123"

    # Error response with code != 0
    error_resp = {
        "code": 40001,
        "msg": "Invalid token",
        "data": None,
    }
    with pytest.raises(APIResponseError) as exc_info:
        unwrap_biz_data(error_resp)
    assert exc_info.value.code == 40001
    assert "Invalid token" in str(exc_info.value)


def test_conversation_id_encoding_decoding():
    # Full id with message
    cid = encode_conversation_id("sess_abc", 42)
    assert cid == "sess_abc:42"

    sess_id, msg_id = decode_conversation_id(cid)
    assert sess_id == "sess_abc"
    assert msg_id == 42

    # None / empty handling
    assert decode_conversation_id(None) == (None, None)
    assert decode_conversation_id("") == (None, None)

    # Session only
    cid_no_msg = encode_conversation_id("sess_xyz", None)
    assert cid_no_msg == "sess_xyz"
    s_only, m_only = decode_conversation_id(cid_no_msg)
    assert s_only == "sess_xyz"
    assert m_only is None


def test_resolve_model_parameters():
    # deepseek-chat -> model_type=default, thinking=False
    m_type, thinking, search = resolve_model_parameters("deepseek-chat")
    assert m_type == "default"
    assert thinking is False
    assert search is False

    # deepseek-reasoner -> model_type=default, thinking=True
    m_type, thinking, search = resolve_model_parameters("deepseek-reasoner")
    assert m_type == "default"
    assert thinking is True

    # Override thinking flag
    m_type, thinking, search = resolve_model_parameters("deepseek-chat", thinking=True, search=True)
    assert thinking is True
    assert search is True


def test_parse_sse_stream_frames():
    state = SSEParserState()

    # 1. Snapshot frame with initial content and message_id
    snapshot_json = (
        'data: {"v":{"response":{"message_id":999,"fragments":[{"type":"RESPONSE","content":"Hello"}]}}}'
    )
    chunk1 = parse_sse_line(snapshot_json, state)
    assert chunk1 is not None
    assert chunk1.text == "Hello"
    assert chunk1.message_id == 999
    assert state.accumulated_content == "Hello"

    # 2. Append frame setting path
    path_frame = 'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" world"}'
    chunk2 = parse_sse_line(path_frame, state)
    assert chunk2 is not None
    assert chunk2.text == " world"
    assert not chunk2.is_thinking
    assert state.accumulated_content == "Hello world"

    # 3. Bare append frame
    bare_frame = 'data: {"v":"!"}'
    chunk3 = parse_sse_line(bare_frame, state)
    assert chunk3 is not None
    assert chunk3.text == "!"
    assert state.accumulated_content == "Hello world!"

    # 4. Thinking frame
    thinking_path_frame = 'data: {"p":"response/fragments/-1/thinking_content","o":"APPEND","v":"Step 1..."}'
    chunk4 = parse_sse_line(thinking_path_frame, state)
    assert chunk4 is not None
    assert chunk4.reasoning_content == "Step 1..."
    assert chunk4.is_thinking
    assert state.accumulated_thinking == "Step 1..."

    # 5. [DONE] frame
    done_frame = 'data: [DONE]'
    assert parse_sse_line(done_frame, state) is None
