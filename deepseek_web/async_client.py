"""
Asynchronous DeepSeek Web Client.

Fully non-blocking client using httpx.AsyncClient for high-concurrency applications,
bots, and async web servers.
"""

from __future__ import annotations

import asyncio
import io
import mimetypes
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from .auth import get_session
from .models import (
    APIResponseError,
    ChatReply,
    ChatSessionItem,
    ChatStreamChunk,
    DEFAULT_MODEL,
    FileUploadResult,
    PoWChallengeError,
    RateLimitError,
    Session,
)
from .pow import DeepSeekPow
from .protocol import (
    BASE_URL,
    COMPLETION_PATH,
    FILE_FETCH_PATH,
    FILE_FORK_PATH,
    FILE_UPLOAD_PATH,
    POW_CHALLENGE_PATH,
    SESSION_CREATE_PATH,
    SESSION_DELETE_PATH,
    SESSION_FETCH_PAGE_PATH,
    USER_PROFILE_PATH,
    build_base_headers,
    decode_conversation_id,
    encode_conversation_id,
    parse_sse_line,
    resolve_model_parameters,
    unwrap_biz_data,
    SSEParserState,
)


class AsyncStreamResponse:
    """Asynchronous completion stream wrapper."""

    def __init__(
        self,
        client: AsyncDeepSeekClient,
        prompt: str,
        session_id: str,
        parent_id: Optional[int],
        model_type: Optional[str],
        thinking: bool,
        search: bool,
        ref_file_ids: Optional[List[str]] = None,
    ):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model_type = model_type
        self._thinking = thinking
        self._search = search
        self._ref_file_ids = ref_file_ids or []
        self._state = SSEParserState(session_id=session_id)
        self._consumed = False

    def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
        return self._iter_stream()

    async def _iter_stream(self) -> AsyncIterator[ChatStreamChunk]:
        if self._consumed:
            raise RuntimeError("Async stream has already been consumed.")
        self._consumed = True

        pow_header = await self._client._get_pow_header(COMPLETION_PATH)

        body: Dict[str, Any] = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": self._ref_file_ids,
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        if self._model_type is not None:
            body["model_type"] = self._model_type

        headers = {"x-ds-pow-response": pow_header}

        async with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            if resp.status_code == 429:
                raise RateLimitError("DeepSeek completion rate limit reached. Please wait before retrying.")
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line, self._state)
                if chunk is not None:
                    yield chunk

    @property
    def conversation_id(self) -> str:
        return encode_conversation_id(self._session_id, self._state.message_id)

    @property
    def full_text(self) -> str:
        return self._state.accumulated_content

    @property
    def reasoning_content(self) -> str:
        return self._state.accumulated_thinking

    @property
    def reply(self) -> ChatReply:
        return ChatReply(
            text=self.full_text,
            reasoning_content=self.reasoning_content,
            conversation_id=self.conversation_id,
            chat_session_id=self._session_id,
            message_id=self._state.message_id,
        )


class AsyncDeepSeekClient:
    """Asynchronous client for DeepSeek web application."""

    def __init__(
        self,
        session: Optional[Session] = None,
        token: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None,
        pow_solver: Optional[DeepSeekPow] = None,
        timeout: float = 120.0,
        allow_interactive: bool = False,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        if session:
            self.session = session
        else:
            self.session = get_session(
                token=token,
                cookies=cookies,
                user_agent=user_agent,
                allow_interactive=allow_interactive,
            )

        self._pow = pow_solver or DeepSeekPow()

        if http_client:
            self._http = http_client
        else:
            self._http = httpx.AsyncClient(
                base_url=BASE_URL,
                headers=build_base_headers(self.session),
                cookies=self.session.cookies,
                timeout=httpx.Timeout(timeout, read=300.0),
                follow_redirects=True,
            )

    async def __aenter__(self) -> "AsyncDeepSeekClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()

    # -------------------------------------------------------------------------
    # Internal PoW Helper
    # -------------------------------------------------------------------------

    async def _get_pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        """Request and solve PoW challenge asynchronously offloading solve to thread."""
        try:
            resp = await self._http.post(POW_CHALLENGE_PATH, json={"target_path": target_path})
            if resp.status_code == 429:
                raise RateLimitError("Rate limit encountered while requesting PoW challenge.")
            resp.raise_for_status()
            data = resp.json()
            biz = unwrap_biz_data(data)
            challenge = biz.get("challenge")
            if not challenge:
                raise PoWChallengeError(f"No challenge found in PoW response: {data}")

            # Solve CPU-bound PoW in a thread pool to avoid blocking the asyncio loop
            return await asyncio.to_thread(self._pow.make_header, challenge)
        except Exception as e:
            if isinstance(e, (PoWChallengeError, RateLimitError)):
                raise
            raise PoWChallengeError(f"Failed to acquire and solve PoW challenge for {target_path}: {e}") from e

    # -------------------------------------------------------------------------
    # User Profile & Verification
    # -------------------------------------------------------------------------

    async def get_user_profile(self) -> Dict[str, Any]:
        """Fetch current user profile to verify authentication validity."""
        resp = await self._http.get(USER_PROFILE_PATH)
        resp.raise_for_status()
        return unwrap_biz_data(resp.json())

    # -------------------------------------------------------------------------
    # Conversation & Session Management
    # -------------------------------------------------------------------------

    async def create_chat_session(self) -> str:
        """Create a new chat session on DeepSeek and return its session ID."""
        resp = await self._http.post(SESSION_CREATE_PATH, json={})
        resp.raise_for_status()
        biz = unwrap_biz_data(resp.json())
        session_info = biz.get("chat_session") or biz
        session_id = session_info.get("id")
        if not session_id:
            raise APIResponseError(-1, f"Failed to retrieve chat session ID from response: {biz}")
        return str(session_id)

    async def list_chat_sessions(self, count: int = 20) -> List[ChatSessionItem]:
        """Fetch list of user's recent chat sessions."""
        resp = await self._http.get(SESSION_FETCH_PAGE_PATH, params={"count": count})
        resp.raise_for_status()
        biz = unwrap_biz_data(resp.json())
        items: List[ChatSessionItem] = []
        sessions = biz.get("chat_sessions", [])
        if isinstance(sessions, list):
            for s in sessions:
                if isinstance(s, dict) and "id" in s:
                    items.append(
                        ChatSessionItem(
                            id=str(s["id"]),
                            title=s.get("title", "Untitled Session"),
                            created_at=s.get("inserted_at") or s.get("created_at") or 0,
                            updated_at=s.get("updated_at") or 0,
                            raw=s,
                        )
                    )
        return items

    async def delete_chat_session(self, chat_session_id: str) -> bool:
        """Delete an existing chat session by ID."""
        resp = await self._http.post(SESSION_DELETE_PATH, json={"chat_session_id": chat_session_id})
        resp.raise_for_status()
        unwrap_biz_data(resp.json())
        return True

    # -------------------------------------------------------------------------
    # Chat & Streaming
    # -------------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[bool] = None,
        search: Optional[bool] = None,
        ref_file_ids: Optional[List[str]] = None,
    ) -> AsyncStreamResponse:
        """Stream an incremental completion response asynchronously."""
        session_id, parent_id = decode_conversation_id(conversation_id)

        # For async stream, we must resolve session_id if new
        # We handle this via lazy creation or when first token requested
        # To keep interface clean, if session_id is None, create lazily in _iter_stream or helper
        model_type, thinking_enabled, search_enabled = resolve_model_parameters(
            model=model if session_id is None else None,
            thinking=thinking,
            search=search,
        )

        return _LazyAsyncStreamResponse(
            client=self,
            prompt=prompt,
            session_id=session_id,
            parent_id=parent_id,
            model_type=model_type if session_id is None else None,
            thinking=thinking_enabled,
            search=search_enabled,
            ref_file_ids=ref_file_ids,
        )

    async def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[bool] = None,
        search: Optional[bool] = None,
        ref_file_ids: Optional[List[str]] = None,
    ) -> ChatReply:
        """Send a prompt to DeepSeek and await the full reply."""
        stream_resp = self.stream(
            prompt=prompt,
            conversation_id=conversation_id,
            model=model,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )
        async for _ in stream_resp:
            pass
        return stream_resp.reply

    # -------------------------------------------------------------------------
    # File Upload & Attachment
    # -------------------------------------------------------------------------

    async def upload_file(
        self,
        file: Union[str, Path, bytes, io.IOBase],
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> FileUploadResult:
        """Upload a file or document to DeepSeek asynchronously."""
        if isinstance(file, (str, Path)):
            p = Path(file)
            filename = filename or p.name
            file_bytes = p.read_bytes()
        elif isinstance(file, bytes):
            filename = filename or "upload.bin"
            file_bytes = file
        elif hasattr(file, "read"):
            filename = filename or getattr(file, "name", "upload.bin")
            file_bytes = file.read()
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {type(file)}")

        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"

        pow_header = await self._get_pow_header(FILE_UPLOAD_PATH)

        headers = dict(self._http.headers)
        headers.pop("content-type", None)
        headers["x-ds-pow-response"] = pow_header

        files = {"file": (filename, file_bytes, content_type)}

        resp = await self._http.post(FILE_UPLOAD_PATH, files=files, headers=headers)
        resp.raise_for_status()
        biz = unwrap_biz_data(resp.json())
        file_id = biz.get("id") or biz.get("file_id")
        if not file_id:
            raise APIResponseError(-1, f"No file ID in upload response: {biz}")

        return FileUploadResult(
            file_id=str(file_id),
            filename=filename,
            content_type=content_type,
            size=len(file_bytes),
            status="uploaded",
        )

    async def fork_file_task(self, file_id: str, to_model_type: str = "vision") -> str:
        """Fork an uploaded file to a specific task/model type asynchronously."""
        resp = await self._http.post(
            FILE_FORK_PATH,
            json={"file_id": file_id, "to_model_type": to_model_type},
        )
        resp.raise_for_status()
        biz = unwrap_biz_data(resp.json())
        forked_id = biz.get("id") or biz.get("file_id") or file_id
        return str(forked_id)

    async def fetch_file_statuses(self, file_ids: List[str]) -> Dict[str, Any]:
        """Fetch processing status for file IDs asynchronously."""
        if not file_ids:
            return {}
        resp = await self._http.get(FILE_FETCH_PATH, params={"file_ids": file_ids})
        resp.raise_for_status()
        return unwrap_biz_data(resp.json())

    async def wait_for_file_parsing(
        self,
        file_ids: List[str],
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> List[str]:
        """Wait asynchronously for DeepSeek to parse uploaded files."""
        if not file_ids:
            return []
        deadline = time.time() + timeout
        ready_ids: List[str] = []

        while time.time() < deadline:
            statuses = await self.fetch_file_statuses(file_ids)
            all_done = True
            for fid in file_ids:
                finfo = statuses.get(fid, {}) if isinstance(statuses, dict) else {}
                status = str(finfo.get("status", "")).upper()
                if status in ("SUCCESS", "COMPLETED", "DONE"):
                    if fid not in ready_ids:
                        ready_ids.append(fid)
                elif status in ("FAILED", "ERROR", "PARSE_FAILED"):
                    ready_ids.append(fid)
                elif status in ("PENDING", "PARSING", "UPLOADING", "QUEUED"):
                    all_done = False
                else:
                    ready_ids.append(fid)

            if all_done and len(ready_ids) == len(file_ids):
                return ready_ids

            await asyncio.sleep(poll_interval)

        return ready_ids or file_ids

    async def upload_and_prepare_file(
        self,
        file: Union[str, Path, bytes, io.IOBase],
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        for_vision: bool = False,
    ) -> str:
        """Upload, optionally fork, and wait for parsing asynchronously."""
        res = await self.upload_file(file, filename=filename, content_type=content_type)
        fid = res.file_id
        if for_vision:
            fid = await self.fork_file_task(fid, to_model_type="vision")
        await self.wait_for_file_parsing([fid], timeout=15.0)
        return fid


class _LazyAsyncStreamResponse(AsyncStreamResponse):
    """Handles async session creation on first iteration if session_id is None."""

    async def _iter_stream(self) -> AsyncIterator[ChatStreamChunk]:
        if self._session_id is None:
            self._session_id = await self._client.create_chat_session()
            self._state.session_id = self._session_id
        async for chunk in super()._iter_stream():
            yield chunk
