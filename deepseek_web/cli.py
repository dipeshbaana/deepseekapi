"""
Command-line interface (CLI) for DeepSeek Web Client.

Commands:
    login       - Authenticate via browser (Playwright) or interactive token prompt
    chat        - Interactive multi-turn chat in terminal with streaming & reasoning
    sessions    - List or delete remote chat sessions
    upload      - Upload a document or image to DeepSeek
    serve       - Run OpenAI-compatible local API server
    whoami      - Test current credentials against DeepSeek API
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .auth import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_SESSION_FILE,
    get_session,
    login_with_playwright,
    prompt_for_token,
)
from .client import DeepSeekClient
from .models import AVAILABLE_MODELS, DEFAULT_MODEL, AuthenticationError, Session
from .server.app import run_server

# Check if rich is available for enhanced terminal formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_info(msg: str):
    if HAS_RICH:
        console.print(f"[bold cyan]ℹ[/bold cyan] {msg}")
    else:
        print(f"[*] {msg}")


def print_success(msg: str):
    if HAS_RICH:
        console.print(f"[bold green]✔[/bold green] {msg}")
    else:
        print(f"[+] {msg}")


def print_error(msg: str):
    if HAS_RICH:
        console.print(f"[bold red]✖[/bold red] {msg}")
    else:
        print(f"[!] {msg}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Subcommand Handlers
# -----------------------------------------------------------------------------

def cmd_login(args):
    """Handle 'login' subcommand."""
    if args.method == "token":
        session = prompt_for_token()
        print_success(f"Authenticated successfully! Session saved to {DEFAULT_SESSION_FILE}")
        return

    try:
        session = login_with_playwright(
            profile_dir=args.profile_dir,
            session_file=args.session_file,
            headless=args.headless,
            timeout=args.timeout,
        )
        print_success(f"Authenticated successfully! Session saved to {DEFAULT_SESSION_FILE}")
    except Exception as e:
        print_error(f"Browser login failed: {e}")
        if sys.stdin.isatty():
            print_info("Falling back to manual token prompt...")
            prompt_for_token()
        else:
            sys.exit(1)


def cmd_whoami(args):
    """Handle 'whoami' subcommand."""
    try:
        client = DeepSeekClient(allow_interactive=False)
        profile = client.get_user_profile()
        print_success("Session token is valid!")
        if HAS_RICH:
            console.print_json(data=profile)
        else:
            import json
            print(json.dumps(profile, indent=2))
    except AuthenticationError as e:
        print_error(f"Authentication error: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to fetch profile: {e}")
        sys.exit(1)


def cmd_chat(args):
    """Handle 'chat' subcommand for interactive terminal conversation."""
    try:
        client = DeepSeekClient(allow_interactive=True)
    except Exception as e:
        print_error(f"Failed to initialize client: {e}")
        sys.exit(1)

    ref_file_ids = []
    if args.attach:
        p = Path(args.attach)
        if not p.exists():
            print_error(f"Attachment file not found: {p}")
            sys.exit(1)
        print_info(f"Uploading attachment: {p.name}...")
        try:
            fid = client.upload_and_prepare_file(p, for_vision=args.vision)
            ref_file_ids.append(fid)
            print_success(f"File uploaded & attached: {fid}")
        except Exception as e:
            print_error(f"Failed to upload attachment: {e}")
            sys.exit(1)

    model = args.model or DEFAULT_MODEL
    thinking = args.thinking
    search = args.search

    print_info(f"Starting chat session with model='{model}' (thinking={thinking}, search={search})")
    print_info("Type 'exit', 'quit', or Ctrl+C to stop. Type 'clear' to start a new thread.\n")

    conversation_id: Optional[str] = None

    while True:
        try:
            if HAS_RICH:
                prompt = console.input("[bold blue]You>[/bold blue] ").strip()
            else:
                prompt = input("You> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("exit", "quit", "q"):
                print_info("Goodbye!")
                break

            if prompt.lower() == "clear":
                conversation_id = None
                print_info("Cleared conversation thread. Starting fresh turn.")
                continue

            stream_resp = client.stream(
                prompt=prompt,
                conversation_id=conversation_id,
                model=model if conversation_id is None else None,
                thinking=thinking,
                search=search,
                ref_file_ids=ref_file_ids if conversation_id is None else None,
            )

            in_thinking_block = False

            for chunk in stream_resp:
                if chunk.reasoning_content:
                    if not in_thinking_block:
                        in_thinking_block = True
                        if HAS_RICH:
                            console.print("[dim italic]\n--- DeepThink Reasoning ---[/dim italic]")
                        else:
                            print("\n--- DeepThink Reasoning ---")
                    sys.stdout.write(chunk.reasoning_content)
                    sys.stdout.flush()

                if chunk.text:
                    if in_thinking_block:
                        in_thinking_block = False
                        if HAS_RICH:
                            console.print("\n[dim italic]--- Response ---[/dim italic]\n")
                        else:
                            print("\n--- Response ---\n")
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()

            sys.stdout.write("\n\n")
            sys.stdout.flush()

            conversation_id = stream_resp.conversation_id

        except (KeyboardInterrupt, EOFError):
            print("\n")
            print_info("Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print_error(f"Error during completion: {e}")


def cmd_sessions(args):
    """Handle 'sessions' subcommand (list or delete)."""
    try:
        client = DeepSeekClient(allow_interactive=False)
    except Exception as e:
        print_error(f"Client initialization error: {e}")
        sys.exit(1)

    if args.action == "list":
        try:
            items = client.list_chat_sessions(count=args.count)
            print_info(f"Found {len(items)} chat sessions:")
            for item in items:
                print(f"  [{item.id}] {item.title}")
        except Exception as e:
            print_error(f"Failed to list chat sessions: {e}")
            sys.exit(1)

    elif args.action == "delete":
        if not args.session_id:
            print_error("Error: --id <session_id> is required for delete.")
            sys.exit(1)
        try:
            client.delete_chat_session(args.session_id)
            print_success(f"Session {args.session_id} deleted successfully.")
        except Exception as e:
            print_error(f"Failed to delete chat session: {e}")
            sys.exit(1)


def cmd_upload(args):
    """Handle 'upload' subcommand."""
    p = Path(args.file)
    if not p.exists():
        print_error(f"File not found: {p}")
        sys.exit(1)

    try:
        client = DeepSeekClient(allow_interactive=False)
        print_info(f"Uploading {p.name} ({p.stat().st_size} bytes)...")
        fid = client.upload_and_prepare_file(p, for_vision=args.vision)
        print_success(f"File uploaded successfully!")
        print(f"File ID: {fid}")
    except Exception as e:
        print_error(f"Upload failed: {e}")
        sys.exit(1)


def cmd_serve(args):
    """Handle 'serve' subcommand to launch OpenAI-compatible server."""
    run_server(host=args.host, port=args.port)


# -----------------------------------------------------------------------------
# Main Parser
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="deepseek-web",
        description="CLI tool for DeepSeek Web Client & OpenAI-compatible local proxy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # login
    p_login = subparsers.add_parser("login", help="Sign in to DeepSeek")
    p_login.add_argument("--method", choices=["browser", "token"], default="browser", help="Login method")
    p_login.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    p_login.add_argument("--timeout", type=float, default=300.0, help="Login timeout in seconds")
    p_login.add_argument("--profile-dir", type=str, default=str(DEFAULT_PROFILE_DIR), help="Browser profile directory")
    p_login.add_argument("--session-file", type=str, default=str(DEFAULT_SESSION_FILE), help="Session file save location")
    p_login.set_defaults(func=cmd_login)

    # whoami
    p_whoami = subparsers.add_parser("whoami", help="Check current session and user profile")
    p_whoami.set_defaults(func=cmd_whoami)

    # chat
    p_chat = subparsers.add_parser("chat", help="Start interactive terminal chat")
    p_chat.add_argument("-m", "--model", choices=list(AVAILABLE_MODELS.keys()), default=DEFAULT_MODEL, help="Model")
    p_chat.add_argument("--thinking", action="store_true", default=None, help="Enable DeepThink reasoning")
    p_chat.add_argument("--no-thinking", action="store_false", dest="thinking", help="Disable DeepThink reasoning")
    p_chat.add_argument("--search", action="store_true", default=None, help="Enable web search")
    p_chat.add_argument("-a", "--attach", type=str, help="Attach file to initial conversation turn")
    p_chat.add_argument("--vision", action="store_true", help="Fork attachment for vision model")
    p_chat.set_defaults(func=cmd_chat)

    # sessions
    p_sessions = subparsers.add_parser("sessions", help="Manage chat sessions")
    p_sessions.add_argument("action", choices=["list", "delete"], help="Action: list or delete")
    p_sessions.add_argument("--count", type=int, default=20, help="Number of sessions to list")
    p_sessions.add_argument("--id", dest="session_id", type=str, help="Chat session ID to delete")
    p_sessions.set_defaults(func=cmd_sessions)

    # upload
    p_upload = subparsers.add_parser("upload", help="Upload a file to DeepSeek")
    p_upload.add_argument("file", type=str, help="Path of file to upload")
    p_upload.add_argument("--vision", action="store_true", help="Fork uploaded file for vision model")
    p_upload.set_defaults(func=cmd_upload)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run local OpenAI-compatible API server")
    p_serve.add_argument("--host", type=str, default=None, help="Server host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None, help="Server port (default: 8000)")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
