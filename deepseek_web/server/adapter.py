"""
Adapter utilities to convert between OpenAI API format and DeepSeek web protocol.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Tuple

from ..models import ChatReply, ChatStreamChunk
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoiceDelta,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Format an OpenAI messages list into prompt text suitable for DeepSeek.

    If single message: returns its content directly.
    If multiple messages: formats system instructions and conversation context.
    """
    if not messages:
        return ""

    if len(messages) == 1:
        content = messages[0].content
        if isinstance(content, str):
            return content
        return json.dumps(content)

    formatted_parts: List[str] = []
    for msg in messages:
        role = msg.role.capitalize()
        content = msg.content
        if isinstance(content, list):
            # Extract text from list of content blocks
            text_blocks = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text_blocks.append(b.get("text", ""))
                else:
                    text_blocks.append(str(b))
            content_str = " ".join(text_blocks)
        else:
            content_str = str(content)

        formatted_parts.append(f"{role}: {content_str}")

    return "\n\n".join(formatted_parts)


def format_chunk(
    chunk_id: str,
    model: str,
    text_delta: str = "",
    reasoning_delta: str = "",
    role: str = None,
    finish_reason: str = None,
    conversation_id: str = None,
) -> str:
    """Serialize a completion chunk into OpenAI SSE line: data: {...}\n\n."""
    delta = ChatCompletionChoiceDelta()
    if role:
        delta.role = role
    if text_delta:
        delta.content = text_delta
    if reasoning_delta:
        delta.reasoning_content = reasoning_delta

    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
        conversation_id=conversation_id,
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def make_completion_response(
    model: str,
    reply: ChatReply,
    prompt: str,
) -> ChatCompletionResponse:
    """Build a non-streaming OpenAI-compatible ChatCompletionResponse."""
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(reply.text) // 4)

    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": reply.text,
    }
    if reply.reasoning_content:
        msg["reasoning_content"] = reply.reasoning_content

    return ChatCompletionResponse(
        id=req_id,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=msg,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        conversation_id=reply.conversation_id,
    )
