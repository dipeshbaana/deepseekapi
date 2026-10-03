"""
DeepSeek Web Client: A clean Python client and OpenAI-compatible proxy for chat.deepseek.com.
"""

from .async_client import AsyncDeepSeekClient, AsyncStreamResponse
from .auth import get_session, login_with_playwright, prompt_for_token
from .client import DeepSeekClient, StreamResponse
from .models import (
    APIResponseError,
    AuthenticationError,
    AVAILABLE_MODELS,
    ChatReply,
    ChatSessionItem,
    ChatStreamChunk,
    DEFAULT_MODEL,
    DeepSeekError,
    FileUploadResult,
    ModelOption,
    ModelType,
    PoWChallengeError,
    RateLimitError,
    Session,
)
from .pow import DeepSeekPow

__version__ = "0.1.0"

__all__ = [
    "DeepSeekClient",
    "AsyncDeepSeekClient",
    "StreamResponse",
    "AsyncStreamResponse",
    "Session",
    "DeepSeekPow",
    "get_session",
    "login_with_playwright",
    "prompt_for_token",
    "ChatReply",
    "ChatStreamChunk",
    "ChatSessionItem",
    "FileUploadResult",
    "ModelOption",
    "ModelType",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DeepSeekError",
    "AuthenticationError",
    "PoWChallengeError",
    "APIResponseError",
    "RateLimitError",
]
