"""
Proof-of-work (PoW) solver for DeepSeek's web application.

DeepSeek chat.deepseek.com requires an `x-ds-pow-response` header for endpoints
such as `/api/v0/chat/completion` and `/api/v0/file/upload_file`.
The challenge uses the 'DeepSeekHashV1' algorithm implemented inside DeepSeek's
official WebAssembly module (sha3_wasm_bg.wasm).

This module runs the WebAssembly binary via wasmtime, with fallback to Node.js
and pure-Python sha3-256 computation if wasmtime is not installed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .models import PoWChallengeError

logger = logging.getLogger(__name__)

DEFAULT_WASM_PATH = Path(__file__).resolve().parent / "assets" / "sha3_wasm_bg.wasm"


class DeepSeekPow:
    """Solves DeepSeek PoW challenges and generates x-ds-pow-response headers."""

    def __init__(self, wasm_path: Optional[Path | str] = None):
        self.wasm_path = Path(wasm_path) if wasm_path else DEFAULT_WASM_PATH
        self._lock = threading.Lock()
        self._wasmtime_initialized = False
        self._store = None
        self._inst = None
        self._memory = None
        self._solve = None
        self._malloc = None
        self._add_to_stack = None

        # Try to initialize wasmtime if available and wasm file exists
        if self.wasm_path.exists():
            self._try_init_wasmtime()

    def _try_init_wasmtime(self) -> bool:
        """Attempt to load the WASM binary using wasmtime."""
        try:
            import wasmtime

            self._store = wasmtime.Store()
            module = wasmtime.Module.from_file(self._store.engine, str(self.wasm_path))
            self._inst = wasmtime.Instance(self._store, module, [])
            exports = self._inst.exports(self._store)
            self._memory = exports["memory"]
            self._solve = exports["wasm_solve"]
            self._malloc = exports["__wbindgen_export_0"]
            self._add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
            self._wasmtime_initialized = True
            logger.debug("PoW wasmtime engine initialized successfully.")
            return True
        except Exception as e:
            logger.warning("Could not initialize wasmtime PoW engine (%s). Fallbacks will be used.", e)
            self._wasmtime_initialized = False
            return False

    def _write_wasm_str(self, text: str) -> tuple[int, int]:
        """Copy a UTF-8 string into wasm memory; return (ptr, len)."""
        data = text.encode("utf-8")
        ptr = self._malloc(self._store, len(data), 1)
        base = self._memory.data_ptr(self._store)
        for i, b in enumerate(data):
            base[ptr + i] = b
        return ptr, len(data)

    def _solve_wasmtime(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Solve using wasmtime engine."""
        retptr = self._add_to_stack(self._store, -16)
        try:
            c_ptr, c_len = self._write_wasm_str(challenge)
            p_ptr, p_len = self._write_wasm_str(prefix)
            self._solve(
                self._store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty)
            )

            mem = self._memory.data_ptr(self._store)
            status = struct.unpack("<i", bytes(mem[retptr : retptr + 4]))[0]
            value = struct.unpack("<d", bytes(mem[retptr + 8 : retptr + 16]))[0]
            if status == 1:
                return int(value)
            return None
        finally:
            self._add_to_stack(self._store, 16)

    def _solve_node(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Fallback solver using Node.js WebAssembly API if node is installed."""
        if not self.wasm_path.exists():
            return None
        node_script = f"""
const fs = require('fs');
const wasmBuffer = fs.readFileSync({json.dumps(str(self.wasm_path))});
WebAssembly.instantiate(wasmBuffer).then(res => {{
    const exp = res.instance.exports;
    const mem = new Uint8Array(exp.memory.buffer);
    function writeStr(str) {{
        const b = Buffer.from(str, 'utf8');
        const ptr = exp.__wbindgen_export_0(b.length, 1);
        const m = new Uint8Array(exp.memory.buffer);
        m.set(b, ptr);
        return [ptr, b.length];
    }}
    const retptr = exp.__wbindgen_add_to_stack_pointer(-16);
    try {{
        const [cPtr, cLen] = writeStr({json.dumps(challenge)});
        const [pPtr, pLen] = writeStr({json.dumps(prefix)});
        exp.wasm_solve(retptr, cPtr, cLen, pPtr, pLen, {float(difficulty)});
        const view = new DataView(exp.memory.buffer);
        const status = view.getInt32(retptr, true);
        const value = view.getFloat64(retptr + 8, true);
        if (status === 1) {{
            process.stdout.write(Math.floor(value).toString());
        }} else {{
            process.exit(1);
        }}
    }} finally {{
        exp.__wbindgen_add_to_stack_pointer(16);
    }}
}}).catch(err => {{
    process.stderr.write(err.toString());
    process.exit(1);
}});
"""
        try:
            res = subprocess.run(
                ["node", "-e", node_script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if res.returncode == 0 and res.stdout.strip().isdigit():
                return int(res.stdout.strip())
        except Exception as e:
            logger.debug("Node.js PoW fallback error: %s", e)
        return None

    def _solve_pure_python(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Pure-Python fallback using SHA3-256 for environments without WASM runtime."""
        try:
            diff = int(difficulty)
            if diff <= 0:
                return 0
            threshold = (2**32) // diff
            max_iter = min(diff * 4, 2_000_000)

            for nonce in range(max_iter):
                data = (challenge + prefix + str(nonce)).encode("utf-8")
                h = hashlib.sha3_256(data).digest()
                val = struct.unpack("<I", h[:4])[0]
                if val < threshold:
                    return nonce
            return None
        except Exception as e:
            logger.debug("Pure python PoW fallback error: %s", e)
            return None

    def solve(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Thread-safe method to solve a PoW challenge using the best available method."""
        with self._lock:
            # 1. Try wasmtime
            if self._wasmtime_initialized:
                try:
                    ans = self._solve_wasmtime(challenge, prefix, difficulty)
                    if ans is not None:
                        return ans
                except Exception as e:
                    logger.warning("Wasmtime solve threw error: %s; trying fallback", e)

            # 2. Try Node.js WASM bridge
            ans = self._solve_node(challenge, prefix, difficulty)
            if ans is not None:
                return ans

            # 3. Pure Python fallback
            logger.info("Using pure Python PoW fallback...")
            return self._solve_pure_python(challenge, prefix, difficulty)

    def make_header(self, challenge_dict: Dict[str, Any]) -> str:
        """Construct the base64-encoded `x-ds-pow-response` header from a challenge dictionary.

        `challenge_dict` is the `biz_data.challenge` object from
        `POST /api/v0/chat/create_pow_challenge`.
        """
        for field in ("algorithm", "challenge", "salt", "expire_at", "difficulty", "signature", "target_path"):
            if field not in challenge_dict:
                raise PoWChallengeError(f"Missing required field '{field}' in PoW challenge: {challenge_dict}")

        prefix = f"{challenge_dict['salt']}_{challenge_dict['expire_at']}_"
        answer = self.solve(
            challenge=str(challenge_dict["challenge"]),
            prefix=prefix,
            difficulty=float(challenge_dict["difficulty"]),
        )

        if answer is None:
            raise PoWChallengeError(
                f"Failed to solve PoW challenge for target {challenge_dict.get('target_path')}"
            )

        payload = {
            "algorithm": challenge_dict["algorithm"],
            "challenge": challenge_dict["challenge"],
            "salt": challenge_dict["salt"],
            "answer": answer,
            "signature": challenge_dict["signature"],
            "target_path": challenge_dict["target_path"],
        }
        compact_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(compact_json).decode("utf-8")
