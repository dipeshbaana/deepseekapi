"""
FastAPI application providing an OpenAI-compatible HTTP interface for DeepSeek Web Client.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..async_client import AsyncDeepSeekClient
from ..models import (
    AVAILABLE_MODELS,
    AuthenticationError,
    DeepSeekError,
    RateLimitError,
)
from .adapter import format_chunk, make_completion_response, messages_to_prompt
from .schemas import (
    ChatCompletionRequest,
    FileObjectResponse,
    ModelInfo,
    ModelListResponse,
)

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepSeek Web OpenAI-Compatible API",
    description="Local OpenAI-compatible API bridge powered by authenticated DeepSeek Web client.",
    version="0.1.0",
)

# Enable CORS for web apps (e.g. Next.js, OpenWebUI, LibreChat)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_async_client: Optional[AsyncDeepSeekClient] = None


def set_client(client: Optional[AsyncDeepSeekClient]) -> None:
    """Set or override the shared AsyncDeepSeekClient instance (useful for testing)."""
    global _async_client
    _async_client = client


def get_client() -> AsyncDeepSeekClient:
    """Retrieve or lazily initialize the shared AsyncDeepSeekClient."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncDeepSeekClient(allow_interactive=False)
    return _async_client


def verify_server_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Verify local server API key if configured."""
    expected_key = os.getenv("SERVER_API_KEY")
    if not expected_key:
        return  # No API key required

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Expected Bearer token.",
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    authenticated = False
    try:
        c = get_client()
        if c.session and c.session.token:
            authenticated = True
    except Exception:
        pass
    return {"status": "ok", "authenticated": authenticated}


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models compatible with OpenAI SDK."""
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(id=model_id, owned_by="deepseek")
            for model_id in AVAILABLE_MODELS
        ],
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """OpenAI-compatible chat completions endpoint."""
    verify_server_api_key(authorization)

    if not req.messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Field 'messages' cannot be empty.", "type": "invalid_request_error"}},
        )

    prompt = messages_to_prompt(req.messages)

    try:
        client = get_client()
    except AuthenticationError as e:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": str(e), "type": "authentication_error"}},
        )

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        async def stream_generator():
            try:
                # First chunk with role delta
                yield format_chunk(chunk_id, req.model, role="assistant")

                stream_response = client.stream(
                    prompt=prompt,
                    conversation_id=req.conversation_id,
                    model=req.model,
                    thinking=req.thinking,
                    search=req.search,
                    ref_file_ids=req.ref_file_ids,
                )

                async for chunk in stream_response:
                    if chunk.reasoning_content:
                        yield format_chunk(
                            chunk_id,
                            req.model,
                            reasoning_delta=chunk.reasoning_content,
                            conversation_id=stream_response.conversation_id,
                        )
                    if chunk.text:
                        yield format_chunk(
                            chunk_id,
                            req.model,
                            text_delta=chunk.text,
                            conversation_id=stream_response.conversation_id,
                        )

                # Final finish chunk
                yield format_chunk(
                    chunk_id,
                    req.model,
                    finish_reason="stop",
                    conversation_id=stream_response.conversation_id,
                )
                yield "data: [DONE]\n\n"

            except RateLimitError as e:
                err_chunk = {"error": {"message": str(e), "type": "rate_limit_error"}}
                yield f"data: {err_chunk}\n\ndata: [DONE]\n\n"
            except Exception as e:
                logger.exception("Error in stream generator: %s", e)
                err_chunk = {"error": {"message": str(e), "type": "api_error"}}
                yield f"data: {err_chunk}\n\ndata: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming mode
    try:
        reply = await client.chat(
            prompt=prompt,
            conversation_id=req.conversation_id,
            model=req.model,
            thinking=req.thinking,
            search=req.search,
            ref_file_ids=req.ref_file_ids,
        )
        return make_completion_response(req.model, reply, prompt)
    except RateLimitError as e:
        return JSONResponse(status_code=429, content={"error": {"message": str(e), "type": "rate_limit_error"}})
    except AuthenticationError as e:
        return JSONResponse(status_code=401, content={"error": {"message": str(e), "type": "authentication_error"}})
    except DeepSeekError as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "api_error"}})
    except Exception as e:
        logger.exception("Unexpected error in chat completions: %s", e)
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})


@app.post("/v1/files", response_model=FileObjectResponse)
async def upload_file_endpoint(
    file: UploadFile = File(...),
    purpose: str = "assistants",
    authorization: Optional[str] = Header(None),
):
    """OpenAI-compatible file upload endpoint."""
    verify_server_api_key(authorization)

    client = get_client()
    content = await file.read()
    filename = file.filename or "uploaded_file"

    try:
        fid = await client.upload_and_prepare_file(
            file=content,
            filename=filename,
            content_type=file.content_type,
            for_vision=False,
        )
        return FileObjectResponse(
            id=fid,
            bytes=len(content),
            filename=filename,
            purpose=purpose,
            status="processed",
        )
    except Exception as e:
        logger.exception("File upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


def run_server(host: Optional[str] = None, port: Optional[int] = None):
    """Start local uvicorn server."""
    import uvicorn

    server_host = host or os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = int(port or os.getenv("SERVER_PORT", "8000"))

    print(f"Starting DeepSeek OpenAI-compatible server at http://{server_host}:{server_port}")
    print(f"Endpoints: http://{server_host}:{server_port}/v1/chat/completions")
    print(f"Health check: http://{server_host}:{server_port}/healthz")

    uvicorn.run(app, host=server_host, port=server_port, log_level="info")


if __name__ == "__main__":
    run_server()
