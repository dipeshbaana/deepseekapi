"""
Pytest configuration and shared fixtures for deepseek_web tests.
"""

import json
import httpx
import pytest


def create_mock_transport() -> httpx.MockTransport:
    """Create an httpx.MockTransport simulating chat.deepseek.com web endpoints."""

    def handler(request: httpx.Request) -> httpx.Response:
        url_path = request.url.path

        # 1. PoW challenge endpoint
        if url_path == "/api/v0/chat/create_pow_challenge":
            body = json.loads(request.content.decode("utf-8"))
            target_path = body.get("target_path", "/api/v0/chat/completion")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": None,
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "DeepSeekHashV1",
                                "challenge": "b0000b22959bad0cc1ecbbfa07f97191b20332fa10d7341ff9c7ba6e7ed927f1",
                                "salt": "dde3ed472be5a2494ee0",
                                "difficulty": 144000,
                                "expire_at": 1777057596443,
                                "signature": "mock_sig",
                                "target_path": target_path,
                            }
                        }
                    },
                },
            )

        # 2. Session create
        if url_path == "/api/v0/chat_session/create":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "biz_data": {
                            "chat_session": {"id": "mock_session_100"}
                        }
                    },
                },
            )

        # 3. Session fetch page
        if url_path == "/api/v0/chat_session/fetch_page":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "biz_data": {
                            "chat_sessions": [
                                {"id": "mock_session_100", "title": "Session 100", "created_at": 12345}
                            ]
                        }
                    },
                },
            )

        # 4. Session delete
        if url_path == "/api/v0/chat_session/delete":
            return httpx.Response(200, json={"code": 0, "data": {"biz_data": {}}})

        # 5. User profile
        if url_path == "/api/v0/users/current":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "biz_data": {
                            "id": "user_1",
                            "email": "user@example.com",
                            "name": "Test User",
                        }
                    },
                },
            )

        # 6. Chat completion SSE stream
        if url_path == "/api/v0/chat/completion":
            assert "x-ds-pow-response" in request.headers
            sse_content = (
                'data: {"v":{"response":{"message_id":777,"fragments":[{"type":"RESPONSE","content":"Hello"}]}}}\n\n'
                'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" there!"}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse_content,
            )

        # 7. File upload
        if url_path == "/api/v0/file/upload_file":
            assert "x-ds-pow-response" in request.headers
            return httpx.Response(
                200,
                json={"code": 0, "data": {"biz_data": {"id": "file_uploaded_99"}}},
            )

        # 8. File fork
        if url_path == "/api/v0/file/fork_file_task":
            return httpx.Response(
                200,
                json={"code": 0, "data": {"biz_data": {"id": "file_forked_99"}}},
            )

        # 9. File fetch status
        if url_path == "/api/v0/file/fetch_files":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "biz_data": {
                            "file_uploaded_99": {"status": "SUCCESS"},
                            "file_forked_99": {"status": "SUCCESS"},
                        }
                    },
                },
            )

        return httpx.Response(404, json={"detail": "Not found"})

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    return create_mock_transport()
