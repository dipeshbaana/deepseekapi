"""
Data models and exceptions for DeepSeek Web Client.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class DeepSeekError(Exception):
    """Base exception for all DeepSeek web client errors."""
    pass


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails, token is missing, or login required."""
    pass


class PoWChallengeError(DeepSeekError):
    """Raised when solving or submitting Proof-of-Work fails."""
    pass


class APIResponseError(DeepSeekError):
    """Raised when DeepSeek API returns non-zero code or unexpected structure."""
    def __init__(self, code: int, message: str, raw_data: Optional[Dict[str, Any]] = None):
        super().__init__(f"DeepSeek API error [{code}]: {message}")
        self.code = code
        self.message = message
        self.raw_data = raw_data or {}


class RateLimitError(DeepSeekError):
    """Raised when rate limits or anti-abuse controls are encountered."""
    pass


# -----------------------------------------------------------------------------
# Models and Enums
# -----------------------------------------------------------------------------

class ModelType(str, Enum):
    """Internal model_type used by chat.deepseek.com web interface."""
    DEFAULT = "default"  # Fast / Instant model (DeepSeek-V3)
    EXPERT = "expert"    # Expert / deep model (DeepSeek-V3.2 / Expert)


@dataclass
class ModelOption:
    """Mapping of user-facing model identifiers to web wire parameters."""
    id: str
    model_type: str
    thinking_enabled: bool
    description: str


AVAILABLE_MODELS: Dict[str, ModelOption] = {
    "deepseek-chat": ModelOption(
        id="deepseek-chat",
        model_type=ModelType.DEFAULT.value,
        thinking_enabled=False,
        description="DeepSeek-V3 fast conversation model (Instant)",
    ),
    "deepseek-reasoner": ModelOption(
        id="deepseek-reasoner",
        model_type=ModelType.DEFAULT.value,
        thinking_enabled=True,
        description="DeepSeek-R1 reasoning model with DeepThink step-by-step thinking",
    ),
    "deepseek-expert": ModelOption(
        id="deepseek-expert",
        model_type=ModelType.EXPERT.value,
        thinking_enabled=False,
        description="DeepSeek expert model mode",
    ),
    "deepseek-expert-reasoner": ModelOption(
        id="deepseek-expert-reasoner",
        model_type=ModelType.EXPERT.value,
        thinking_enabled=True,
        description="DeepSeek expert model with DeepThink reasoning",
    ),
}

DEFAULT_MODEL = "deepseek-chat"


# -----------------------------------------------------------------------------
# Session Model
# -----------------------------------------------------------------------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class Session:
    """Authenticated session credentials and browser context."""
    token: str
    cookies: Dict[str, str] = field(default_factory=dict)
    user_agent: str = DEFAULT_USER_AGENT
    captured_at: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        """Age of this session in seconds."""
        return max(0.0, time.time() - self.captured_at)

    def is_expired(self, max_age_seconds: float = 6 * 3600) -> bool:
        """Check if session is older than expected TTL (default 6 hours)."""
        return self.age >= max_age_seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            token=data["token"],
            cookies=data.get("cookies", {}),
            user_agent=data.get("user_agent", DEFAULT_USER_AGENT),
            captured_at=data.get("captured_at", time.time()),
        )

    def save(self, path: Path | str) -> None:
        """Save session to disk securely with restricted permissions (0600)."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write to temporary file first then atomic rename with restricted permissions
        content = json.dumps(self.to_dict(), indent=2)
        tmp_file = p.with_suffix(".tmp")
        tmp_file.write_text(content, encoding="utf-8")
        try:
            # Set owner read/write only (chmod 600)
            os.chmod(tmp_file, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Windows may not support POSIX permissions
        tmp_file.replace(p)

    @classmethod
    def load(cls, path: Path | str) -> Optional["Session"]:
        """Load session from path if exists and valid JSON."""
        p = Path(path).resolve()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("token"):
                return None
            return cls.from_dict(data)
        except Exception:
            return None


# -----------------------------------------------------------------------------
# Conversation & Chat Response Models
# -----------------------------------------------------------------------------

@dataclass
class ChatStreamChunk:
    """Single incremental chunk from completion stream."""
    text: str = ""
    reasoning_content: str = ""
    is_thinking: bool = False
    message_id: Optional[int] = None
    session_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_content(self) -> bool:
        return bool(self.text or self.reasoning_content)


@dataclass
class ChatReply:
    """Full completed reply from DeepSeek."""
    text: str
    reasoning_content: str = ""
    conversation_id: str = ""
    chat_session_id: Optional[str] = None
    message_id: Optional[int] = None

    def __str__(self) -> str:
        return self.text


@dataclass
class ChatSessionItem:
    """Chat session metadata as returned by /api/v0/chat_session/fetch_page."""
    id: str
    title: str
    created_at: int
    updated_at: int
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileUploadResult:
    """Result of uploading a file via /api/v0/file/upload_file."""
    file_id: str
    filename: str
    content_type: str
    size: int
    forked_id: Optional[str] = None
    status: str = "uploaded"
