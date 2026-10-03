"""
Pydantic schemas for OpenAI-compatible HTTP server.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    conversation_id: Optional[str] = None
    thinking: Optional[bool] = None
    search: Optional[bool] = None
    ref_file_ids: Optional[List[str]] = None


class ChatCompletionChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    conversation_id: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Dict[str, Any]
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    conversation_id: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepseek"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class FileObjectResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int = Field(default_factory=lambda: int(time.time()))
    filename: str
    purpose: str = "assistants"
    status: str = "processed"
