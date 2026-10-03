"""
Unit tests for the Proof-of-Work (PoW) solver.
"""

import base64
import json
import pytest

from deepseek_web.models import PoWChallengeError
from deepseek_web.pow import DeepSeekPow


def test_pow_wasm_solver():
    """Test PoW solver against a known test vector."""
    solver = DeepSeekPow()

    challenge_dict = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "b0000b22959bad0cc1ecbbfa07f97191b20332fa10d7341ff9c7ba6e7ed927f1",
        "salt": "dde3ed472be5a2494ee0",
        "difficulty": 144000,
        "expire_at": 1777057596443,
        "signature": "sig_abc_123",
        "target_path": "/api/v0/chat/completion",
    }

    # Generate header
    header = solver.make_header(challenge_dict)
    assert header is not None
    assert isinstance(header, str)

    # Decode and verify header structure
    decoded_raw = base64.b64decode(header).decode("utf-8")
    payload = json.loads(decoded_raw)

    assert payload["algorithm"] == "DeepSeekHashV1"
    assert payload["challenge"] == challenge_dict["challenge"]
    assert payload["salt"] == challenge_dict["salt"]
    assert payload["signature"] == "sig_abc_123"
    assert payload["target_path"] == "/api/v0/chat/completion"
    assert isinstance(payload["answer"], int)
    # 58033 is the exact nonce for this test vector
    assert payload["answer"] == 58033


def test_pow_missing_field_raises():
    """Verify that a malformed challenge raises PoWChallengeError."""
    solver = DeepSeekPow()
    with pytest.raises(PoWChallengeError):
        solver.make_header({"algorithm": "DeepSeekHashV1"})
