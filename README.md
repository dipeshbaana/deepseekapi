# DeepSeek Web Client (Python)

A clean, robust, and asynchronous Python client and local OpenAI-compatible HTTP proxy for DeepSeek's web application ([chat.deepseek.com](https://chat.deepseek.com)).

Designed for interoperability, research, and testing purposes while strictly adhering to normal client-server protocol behavior and respecting all platform security mechanisms (CAPTCHA, MFA, Cloudflare/AWS WAF, rate limits, and anti-bot controls).

---

## Features

- **Clean Python Architecture**: Provides both synchronous (`DeepSeekClient`) and asynchronous (`AsyncDeepSeekClient`) interfaces using `httpx`.
- **Realistic Authentication Flow**:
  - Direct token authentication via `DEEPSEEK_TOKEN` environment variable or interactive prompt.
  - Interactive browser login via Playwright (`deepseek-web login`): opens a browser window where you can solve human verification (CAPTCHA / Cloudflare / MFA) manually.
  - Persistent Chromium profile to reuse sessions without repetitive re-logins.
  - Secure session storage on disk with restricted POSIX file permissions (`0600`).
  - Strict security: credentials and session tokens are excluded from git and source code.
- **Proof-of-Work (PoW) Protocol Implementation**:
  - Automatically solves DeepSeek's `DeepSeekHashV1` PoW challenges required for `/api/v0/chat/completion` and `/api/v0/file/upload_file`.
  - Executes DeepSeek's official WebAssembly module (`sha3_wasm_bg.wasm`) inside a secure sandbox using `wasmtime`, with automatic fallbacks to Node.js and pure-Python SHA3-256.
- **Chat & Real-Time Streaming**:
  - Real-time Server-Sent Events (SSE) stream parser.
  - Native separation of **DeepThink reasoning content** (`reasoning_content`) and actual response text (`content`).
  - Thread resumption and multi-turn conversation tracking (`<chat_session_id>:<last_message_id>`).
- **Model Selection & Capabilities**:
  - `deepseek-chat` (DeepSeek-V3 fast instant model)
  - `deepseek-reasoner` (DeepSeek-R1 with DeepThink step-by-step thinking)
  - `deepseek-expert` and `deepseek-expert-reasoner` modes
  - Web search toggle (`search=True`)
- **File Upload & Attachment Support**:
  - Upload documents and images via DeepSeek's internal `/api/v0/file/upload_file` endpoint.
  - Task forking (`/api/v0/file/fork_file_task`) and parse status polling (`/api/v0/file/fetch_files`).
  - Reference uploaded files in chat queries using `ref_file_ids`.
- **OpenAI-Compatible Local Endpoint**:
  - FastAPI server offering `/v1/chat/completions` (streaming & non-streaming) and `/v1/models`.
  - Drop-in replacement for OpenAI SDKs, OpenWebUI, LibreChat, and AI editor extensions.
- **CLI Utility**:
  - `deepseek-web login`, `deepseek-web chat`, `deepseek-web sessions`, `deepseek-web upload`, `deepseek-web whoami`, and `deepseek-web serve`.

---

## Status and Verification

| Functionality | Status | Details |
| :--- | :--- | :--- |
| **PoW Engine (`DeepSeekPow`)** | **Confirmed Working** | Solves `DeepSeekHashV1` using verified WASM test vectors in `<100ms`. |
| **Session Model & Serialization** | **Confirmed Working** | Secure atomic saving with `0600` permissions and expiry detection. |
| **SSE Stream Parser** | **Confirmed Working** | Handles snapshot frames, fragment append frames, and reasoning separation. |
| **OpenAI Server & Schemas** | **Confirmed Working** | Fully compliant with OpenAI specification for completions, chunks, and models. |
| **Playwright Login Flow** | **Client-Server Dependent** | Requires local graphical display or user interaction for initial CAPTCHA/MFA. |
| **Internal Web Endpoints (`/api/v0/`)** | **Client-Server Dependent** | Subject to DeepSeek's web frontend updates, terms of service, and anti-abuse controls. |

> [!NOTE]
> This project reproduces the normal browser protocol for authenticated accounts that you own. For high-volume production deployments, please use the official developer platform at [api.deepseek.com](https://platform.deepseek.com/).

---

## Installation

### Prerequisites
- Python 3.9+
- Linux, macOS, or Windows

### Step 1: Clone and Install Dependencies

```bash
git clone <repo-url> deepseek
cd deepseek

# Recommended: use a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

### Step 2: Install Playwright Browser (Optional, for browser login)

```bash
playwright install chromium
```

---

## Authentication

You have two primary ways to authenticate:

### Method A: Browser Interactive Login (Recommended if CAPTCHA/MFA is present)

Run the CLI login command:
```bash
deepseek-web login
```
This opens a Chromium window pointing to `https://chat.deepseek.com/sign_in`. Sign in to your account and solve any human verification or MFA challenge interactively. The client automatically detects the resulting session token and cookies, saves them securely to `./session/session.json`, and closes the browser.

### Method B: Environment Variable / DevTools Token

If you already have a signed-in browser session:
1. Open [chat.deepseek.com](https://chat.deepseek.com) in your browser.
2. Press `F12` to open Developer Tools.
3. Go to **Application** (or **Storage**) $\rightarrow$ **Local Storage** $\rightarrow$ `https://chat.deepseek.com`.
4. Copy the value of `userToken` (if it's a JSON string like `{"value":"..."}`, copy the token string inside `value`).
5. Set it in your `.env` file or export it:

```bash
cp .env.example .env
export DEEPSEEK_TOKEN="your_token_here"
```

Or run the interactive prompt:
```bash
deepseek-web login --method token
```

---

## Python API Usage

### 1. Synchronous Chat & Streaming

```python
from deepseek_web import DeepSeekClient

# Initializes session from .env, saved session.json, or prompt
with DeepSeekClient() as client:
    # Non-streaming chat
    reply = client.chat("Explain quantum entanglement in simple terms.")
    print("Response:\n", reply.text)
    if reply.reasoning_content:
        print("\nThinking:\n", reply.reasoning_content)

    # Multi-turn conversation continuation
    follow_up = client.chat(
        prompt="Can you summarize that in one sentence?",
        conversation_id=reply.conversation_id,  # continues previous thread
    )
    print("\nFollow-up:\n", follow_up.text)
```

### 2. Streaming with DeepThink Reasoning (Sync)

```python
from deepseek_web import DeepSeekClient

with DeepSeekClient() as client:
    stream = client.stream(
        prompt="Solve this math puzzle: If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?",
        model="deepseek-reasoner",  # Enables R1 reasoning mode
    )

    for chunk in stream:
        if chunk.reasoning_content:
            print(f"[Think] {chunk.reasoning_content}", end="", flush=True)
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n\nSaved Conversation ID:", stream.conversation_id)
```

### 3. Asynchronous Client (`AsyncDeepSeekClient`)

```python
import asyncio
from deepseek_web import AsyncDeepSeekClient

async def main():
    async with AsyncDeepSeekClient() as client:
        # Check authentication validity
        profile = await client.get_user_profile()
        print(f"Logged in as: {profile.get('email') or profile.get('name')}")

        # Async streaming
        stream = client.stream(
            prompt="Write a Python generator for Fibonacci numbers.",
            model="deepseek-chat",
        )

        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

asyncio.run(main())
```

### 4. File Upload and Attachment

```python
from deepseek_web import DeepSeekClient

with DeepSeekClient() as client:
    # 1. Upload and prepare document
    file_id = client.upload_and_prepare_file(
        file="report.pdf",
        filename="report.pdf",
        for_vision=False,
    )
    print("Uploaded File ID:", file_id)

    # 2. Reference file in conversation
    reply = client.chat(
        prompt="Please summarize key findings in the attached document.",
        ref_file_ids=[file_id],
    )
    print("Analysis:\n", reply.text)
```

---

## Local OpenAI-Compatible Server

You can launch a local HTTP endpoint that translates standard OpenAI API requests to the DeepSeek web interface:

```bash
# Start server on http://127.0.0.1:8000
deepseek-web serve --port 8000
```

### Calling via the Official OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",  # or your SERVER_API_KEY
)

# Streaming with reasoning delta support
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain how Dijkstra's algorithm works."},
    ],
    stream=True,
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

---

## CLI Reference

```bash
# Authenticate
deepseek-web login [--method browser|token] [--headless] [--timeout 300]

# Check authentication profile
deepseek-web whoami

# Interactive terminal chat with live streaming and DeepThink display
deepseek-web chat [-m deepseek-chat|deepseek-reasoner] [--thinking] [--search] [-a <file_path>]

# Manage remote sessions
deepseek-web sessions list [--count 20]
deepseek-web sessions delete --id <session_id>

# Upload a document
deepseek-web upload <file_path> [--vision]

# Start OpenAI-compatible API server
deepseek-web serve [--host 127.0.0.1] [--port 8000]
```

---

## Testing

Run unit and integration tests with pytest:

```bash
pytest -v
```

---

## Security and Compliance

- **Credentials Safety**: Never commit `.env`, `session.json`, or profile directories. All sensitive session files are excluded via `.gitignore`.
- **File Permissions**: Saved session files are written with POSIX mode `0600` (read/write by owner only).
- **Anti-Bot & Rate Limits**: This project does not attempt to bypass CAPTCHAs, MFA, or rate limits. If a 429 status code is received, exponential backoff should be observed.
