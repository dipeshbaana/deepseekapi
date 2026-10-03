"""
Protocol constants, header builders, and SSE stream parser for chat.deepseek.com.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

from .models import (
    APIResponseError,
    ChatStreamChunk,
    DEFAULT_USER_AGENT,
    ModelOption,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    Session,
)

BASE_URL = "https://chat.deepseek.com"

# API Endpoints
COMPLETION_PATH = "/api/v0/chat/completion"
POW_CHALLENGE_PATH = "/api/v0/chat/create_pow_challenge"
SESSION_CREATE_PATH = "/api/v0/chat_session/create"
SESSION_FETCH_PAGE_PATH = "/api/v0/chat_session/fetch_page"
SESSION_DELETE_PATH = "/api/v0/chat_session/delete"
FILE_UPLOAD_PATH = "/api/v0/file/upload_file"
FILE_FORK_PATH = "/api/v0/file/fork_file_task"
FILE_FETCH_PATH = "/api/v0/file/fetch_files"
USER_PROFILE_PATH = "/api/v0/users/current"

CONVERSATION_ID_SEPARATOR = ":"


def build_base_headers(session: Session) -> Dict[str, str]:
    """Construct standard request headers mimicking chat.deepseek.com web client."""
    return {
        "authorization": f"Bearer {session.token}",
        "accept": "*/*",
        "content-type": "application/json",
        "user-agent": session.user_agent or DEFAULT_USER_AGENT,
        "origin": BASE_URL,
        "referer": f"{BASE_URL}/",
        "x-app-version": "2.0.0",
        "x-client-version": "2.0.0",
        "x-client-platform": "web",
        "x-client-locale": "en_US",
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-timezone-offset": "0",
    }


def unwrap_biz_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising APIResponseError on error code."""
    if not isinstance(data, dict):
        raise APIResponseError(-1, f"Expected JSON dictionary, got {type(data).__name__}", raw_data={"raw": data})

    code = data.get("code")
    if code is not None and code != 0:
        msg = data.get("msg") or data.get("message") or f"Unknown error code {code}"
        raise APIResponseError(code, str(msg), raw_data=data)

    inner = data.get("data")
    if isinstance(inner, dict):
        biz = inner.get("biz_data")
        if isinstance(biz, dict):
            return biz
        return inner

    return data


def encode_conversation_id(session_id: str, message_id: Optional[int]) -> str:
    """Encode session_id and message_id into a resume token: <session_id>:<message_id>."""
    if message_id is None:
        return session_id
    return f"{session_id}{CONVERSATION_ID_SEPARATOR}{message_id}"


def decode_conversation_id(conversation_id: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Split conversation_id back into (session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(CONVERSATION_ID_SEPARATOR)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


def resolve_model_parameters(
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    search: Optional[bool] = None,
) -> Tuple[str, bool, bool]:
    """Resolve user-supplied model, thinking flag, and search flag to protocol values.

    Returns (model_type, thinking_enabled, search_enabled).
    """
    model_name = (model or DEFAULT_MODEL).lower()
    option = AVAILABLE_MODELS.get(model_name)

    if option:
        model_type = option.model_type
        thinking_enabled = option.thinking_enabled if thinking is None else thinking
    else:
        # Fallback to default model_type if custom model name
        model_type = "default"
        thinking_enabled = False if thinking is None else thinking

    search_enabled = False if search is None else search
    return model_type, thinking_enabled, search_enabled


@dataclass
class SSEParserState:
    """Tracks parser state across streaming SSE chunks."""
    active_path: Optional[str] = None
    emitted_initial_content: bool = False
    emitted_initial_thinking: bool = False
    message_id: Optional[int] = None
    session_id: Optional[str] = None
    accumulated_content: str = ""
    accumulated_thinking: str = ""


def parse_sse_line(line: str, state: SSEParserState) -> Optional[ChatStreamChunk]:
    """Parse a single SSE line and emit a ChatStreamChunk delta if text or reasoning was appended.

    Handles:
    1. Snapshot frame with full initial response structure
    2. Path updates (`{"p": "response/fragments/-1/content", "o": "APPEND", "v": "..."}`)
    3. Path updates for reasoning (`{"p": "response/fragments/-1/thinking_content", ...}`)
    4. Append frames (`{"v": "delta"}`)
    5. Message ID capture
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None

    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None

    v = obj.get("v")

    # Snapshot frame: full response object
    if isinstance(v, dict) and "response" in v:
        resp = v["response"]
        # Extract message_id
        mid = resp.get("message_id") or resp.get("id") or v.get("message_id") or v.get("id")
        if isinstance(mid, int):
            state.message_id = mid

        chunk_text = ""
        chunk_thinking = ""
        for frag in resp.get("fragments", []):
            ftype = frag.get("type")
            content = frag.get("content", "")
            thinking_content = frag.get("thinking_content", "")

            if ftype == "RESPONSE" and content and not state.emitted_initial_content:
                state.emitted_initial_content = True
                state.active_path = "response/fragments/-1/content"
                chunk_text = content
                state.accumulated_content += content

            if thinking_content and not state.emitted_initial_thinking:
                state.emitted_initial_thinking = True
                state.active_path = "response/fragments/-1/thinking_content"
                chunk_thinking = thinking_content
                state.accumulated_thinking += thinking_content

        if chunk_text or chunk_thinking:
            return ChatStreamChunk(
                text=chunk_text,
                reasoning_content=chunk_thinking,
                is_thinking=bool(chunk_thinking and not chunk_text),
                message_id=state.message_id,
                session_id=state.session_id,
                raw=obj,
            )
        return None

    # Path-setting frame
    if "p" in obj:
        state.active_path = obj["p"]
        if state.active_path.endswith("message_id") and isinstance(v, int):
            state.message_id = v

        if obj.get("o") == "APPEND" and isinstance(v, str):
            if state.active_path.endswith("thinking_content"):
                state.accumulated_thinking += v
                return ChatStreamChunk(
                    text="",
                    reasoning_content=v,
                    is_thinking=True,
                    message_id=state.message_id,
                    session_id=state.session_id,
                    raw=obj,
                )
            elif state.active_path.endswith("content"):
                state.accumulated_content += v
                return ChatStreamChunk(
                    text=v,
                    reasoning_content="",
                    is_thinking=False,
                    message_id=state.message_id,
                    session_id=state.session_id,
                    raw=obj,
                )
        return None

    # Bare append frame to currently active path
    if isinstance(v, str) and state.active_path:
        if state.active_path.endswith("thinking_content"):
            state.accumulated_thinking += v
            return ChatStreamChunk(
                text="",
                reasoning_content=v,
                is_thinking=True,
                message_id=state.message_id,
                session_id=state.session_id,
                raw=obj,
            )
        elif state.active_path.endswith("content"):
            state.accumulated_content += v
            return ChatStreamChunk(
                text=v,
                reasoning_content="",
                is_thinking=False,
                message_id=state.message_id,
                session_id=state.session_id,
                raw=obj,
            )

    return None
