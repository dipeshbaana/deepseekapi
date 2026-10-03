"""
Authentication module for DeepSeek Web Client.

Supports:
1. Direct token via constructor or environment variables (DEEPSEEK_TOKEN / DEEPSEEK_USER_TOKEN)
2. Interactive terminal prompt for Bearer token/cookies
3. Interactive browser-based login flow via Playwright with persistent Chromium profile
4. Secure session serialization with 0600 file permissions
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .models import AuthenticationError, DEFAULT_USER_AGENT, Session

load_dotenv()

logger = logging.getLogger(__name__)

# Default locations
ROOT_DIR = Path.cwd()
DEFAULT_SESSION_FILE = Path(os.getenv("DEEPSEEK_SESSION_FILE", ROOT_DIR / "session" / "session.json"))
DEFAULT_PROFILE_DIR = Path(os.getenv("DEEPSEEK_PROFILE_DIR", ROOT_DIR / "session" / "profile"))

CHAT_URL = "https://chat.deepseek.com/"
SIGNIN_URL = "https://chat.deepseek.com/sign_in"

# Playwright launch arguments to avoid detection flag
BROWSER_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-setuid-sandbox",
]

# Max age for a saved session before headless refresh or re-verification (6 hours)
DEFAULT_MAX_AGE_SECONDS = 6 * 3600

# JS helper to read token from localStorage:
# format in chat.deepseek.com: localStorage.userToken = '{"value":"<TOKEN>","__version":"0"}' or raw string
_READ_TOKEN_JS = """
() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && parsed.value) {
        return parsed.value;
      }
    } catch (_) {}
    return raw;
  } catch (e) {
    return null;
  }
}
"""


def _safe_evaluate(page, js: str) -> Any:
    """Evaluate JS on Playwright page, ignoring transient context destruction during navigation."""
    try:
        return page.evaluate(js)
    except Exception as e:
        msg = str(e).lower()
        if "execution context was destroyed" in msg or "navigation" in msg:
            return None
        raise


def _capture_from_context(context, page) -> Optional[Session]:
    """Extract token, cookies, and user agent from active browser context."""
    token = _safe_evaluate(page, _READ_TOKEN_JS)
    if not token:
        return None
    cookies_list = context.cookies()
    cookies = {c["name"]: c["value"] for c in cookies_list if "name" in c and "value" in c}
    ua = _safe_evaluate(page, "() => navigator.userAgent") or DEFAULT_USER_AGENT
    return Session(
        token=str(token).strip(),
        cookies=cookies,
        user_agent=ua,
        captured_at=time.time(),
    )


def _wait_for_token(page, timeout: float = 300.0) -> Optional[str]:
    """Poll localStorage for userToken until present or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        token = _safe_evaluate(page, _READ_TOKEN_JS)
        if token:
            return str(token).strip()
        page.wait_for_timeout(1000)
    return None


def login_with_playwright(
    profile_dir: Optional[Path | str] = None,
    session_file: Optional[Path | str] = None,
    headless: Optional[bool] = None,
    timeout: float = 300.0,
) -> Session:
    """Launch interactive Playwright browser session for the user to sign in.

    The user completes normal authentication interactively (including any
    CAPTCHA, MFA, email code, or Cloudflare/AWS WAF verification), and the
    resulting session token and cookies are saved to disk.
    """
    from playwright.sync_api import sync_playwright

    prof_dir = Path(profile_dir) if profile_dir else DEFAULT_PROFILE_DIR
    sess_file = Path(session_file) if session_file else DEFAULT_SESSION_FILE
    prof_dir.mkdir(parents=True, exist_ok=True)

    if headless is None:
        headless_env = os.getenv("DEEPSEEK_BROWSER_HEADLESS", "").lower()
        # Default to headless=False unless explicitly requested or no DISPLAY available
        if headless_env in ("true", "1", "yes"):
            headless = True
        else:
            # If no DISPLAY on Linux, warn user
            has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
            headless = not has_display

    print("=" * 60)
    print("DeepSeek Web Authentication via Browser")
    print("=" * 60)
    if headless:
        print("[Auth] Launching browser in HEADLESS mode.")
        print("[Auth] Note: If human verification (CAPTCHA) is required, non-headless mode is recommended.")
    else:
        print("[Auth] Launching browser window. Please complete sign-in and any verifications in the window.")
    print(f"[Auth] Target: {SIGNIN_URL}")
    print(f"[Auth] Waiting for authentication (up to {int(timeout)} seconds)...")

    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(prof_dir),
                headless=headless,
                channel="chrome",
                args=BROWSER_LAUNCH_ARGS,
            )
        except Exception:
            context = p.chromium.launch_persistent_context(
                str(prof_dir),
                headless=headless,
                args=BROWSER_LAUNCH_ARGS,
            )

        page = context.pages[0] if context.pages else context.new_page()

        try:
            page.goto(SIGNIN_URL, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning("Initial navigation encountered exception (%s); continuing...", e)

        token = _wait_for_token(page, timeout=timeout)
        if not token:
            context.close()
            raise AuthenticationError(
                f"Browser login timed out after {int(timeout)} seconds: no session token detected."
            )

        session = _capture_from_context(context, page)
        context.close()

    if not session or not session.token:
        raise AuthenticationError("Logged in successfully but could not extract user token from browser context.")

    session.save(sess_file)
    print(f"[Auth] Login successful! Session saved to {sess_file}")
    return session


def _headless_refresh(profile_dir: Path) -> Optional[Session]:
    """Attempt to capture a fresh token headlessly from the persistent browser profile."""
    if not profile_dir.exists():
        return None
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            try:
                context = p.chromium.launch_persistent_context(
                    str(profile_dir),
                    headless=True,
                    channel="chrome",
                    args=BROWSER_LAUNCH_ARGS,
                )
            except Exception:
                context = p.chromium.launch_persistent_context(
                    str(profile_dir),
                    headless=True,
                    args=BROWSER_LAUNCH_ARGS,
                )
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(CHAT_URL, wait_until="domcontentloaded", timeout=30000)
                session = _capture_from_context(context, page)
            finally:
                context.close()
        return session
    except Exception as e:
        logger.debug("Headless refresh failed: %s", e)
        return None


def prompt_for_token() -> Session:
    """Interactively prompt user in terminal for token and cookies."""
    print("=" * 60)
    print("Interactive DeepSeek Token Configuration")
    print("=" * 60)
    print("To obtain your token from an authenticated browser session:")
    print("1. Open https://chat.deepseek.com in your browser and sign in.")
    print("2. Open Developer Tools (F12 or right-click -> Inspect).")
    print("3. Go to Application / Storage -> Local Storage -> https://chat.deepseek.com")
    print("4. Find the key 'userToken'. Copy its value (or the 'value' field inside it).")
    print("=" * 60)

    token = getpass.getpass("Enter your DeepSeek userToken (input hidden): ").strip()
    if not token:
        # Fallback to standard input if getpass returned empty or piped
        print("Empty input. Enter token: ", end="", flush=True)
        token = sys.stdin.readline().strip()

    # If the user pasted the entire JSON string `{"value":"...", "__version":"0"}`
    if token.startswith("{") and "value" in token:
        try:
            parsed = json.loads(token)
            if isinstance(parsed, dict) and "value" in parsed:
                token = parsed["value"]
        except Exception:
            pass

    if not token:
        raise AuthenticationError("No token provided.")

    cookies_input = input("Enter optional cookies as JSON (press Enter to skip): ").strip()
    cookies = {}
    if cookies_input:
        try:
            cookies = json.loads(cookies_input)
        except Exception:
            # Maybe standard cookie header format "name=val; name2=val2"
            for item in cookies_input.split(";"):
                if "=" in item:
                    k, v = item.strip().split("=", 1)
                    cookies[k.strip()] = v.strip()

    session = Session(token=token, cookies=cookies)
    session.save(DEFAULT_SESSION_FILE)
    print(f"[Auth] Token configured and saved to {DEFAULT_SESSION_FILE}")
    return session


def get_session(
    token: Optional[str] = None,
    cookies: Optional[Dict[str, str]] = None,
    user_agent: Optional[str] = None,
    session_file: Optional[Path | str] = None,
    profile_dir: Optional[Path | str] = None,
    allow_interactive: bool = True,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
) -> Session:
    """Resolve an authenticated Session through the following precedence:

    1. Explicit `token` argument passed by caller
    2. Environment variable `DEEPSEEK_TOKEN` or `DEEPSEEK_USER_TOKEN`
    3. Saved session file (e.g. session/session.json) if not expired
    4. Headless refresh using existing persistent Chromium profile
    5. Interactive browser login (Playwright) if `allow_interactive` is True and display available
    6. Interactive terminal prompt if `allow_interactive` is True and terminal is a TTY
    7. Raises `AuthenticationError` if no valid credentials found.
    """
    # 1. Explicit argument
    if token:
        return Session(
            token=token.strip(),
            cookies=cookies or {},
            user_agent=user_agent or DEFAULT_USER_AGENT,
        )

    # 2. Environment variables
    env_token = os.getenv("DEEPSEEK_TOKEN") or os.getenv("DEEPSEEK_USER_TOKEN")
    if env_token:
        env_cookies: Dict[str, str] = {}
        cookies_str = os.getenv("DEEPSEEK_COOKIES")
        if cookies_str:
            try:
                env_cookies = json.loads(cookies_str)
            except Exception:
                for item in cookies_str.split(";"):
                    if "=" in item:
                        k, v = item.strip().split("=", 1)
                        env_cookies[k.strip()] = v.strip()

        env_ua = os.getenv("DEEPSEEK_USER_AGENT", DEFAULT_USER_AGENT)
        return Session(
            token=env_token.strip(),
            cookies=env_cookies,
            user_agent=env_ua,
        )

    sess_file = Path(session_file) if session_file else DEFAULT_SESSION_FILE
    prof_dir = Path(profile_dir) if profile_dir else DEFAULT_PROFILE_DIR

    # 3. Saved session file
    cached = Session.load(sess_file)
    if cached and not cached.is_expired(max_age_seconds=max_age_seconds):
        logger.debug("Loaded valid cached session from %s (age: %.1fs)", sess_file, cached.age)
        return cached

    # 4. Headless refresh via persistent profile
    if prof_dir.exists():
        logger.debug("Attempting headless refresh from browser profile at %s", prof_dir)
        refreshed = _headless_refresh(prof_dir)
        if refreshed:
            refreshed.save(sess_file)
            return refreshed

    # If non-interactive, fail fast with helpful instruction
    if not allow_interactive:
        raise AuthenticationError(
            "No authenticated DeepSeek session found and interactive login is disabled.\n"
            "To authenticate, please perform one of the following:\n"
            "  1. Set the DEEPSEEK_TOKEN environment variable in your .env or shell\n"
            "  2. Run 'deepseek-web login' to sign in via browser or interactive prompt\n"
            "  3. Save a session JSON file at " + str(sess_file)
        )

    # 5. Interactive login
    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    if has_display:
        try:
            return login_with_playwright(
                profile_dir=prof_dir,
                session_file=sess_file,
                headless=False,
            )
        except Exception as e:
            logger.warning("Browser-based login failed or was cancelled: %s", e)
            if sys.stdin.isatty():
                print("[Auth] Falling back to interactive terminal prompt...")
                return prompt_for_token()
            raise

    # 6. Terminal prompt if in a TTY without GUI display
    if sys.stdin.isatty():
        return prompt_for_token()

    # 7. In headless container with no TTY
    raise AuthenticationError(
        "Cannot complete interactive authentication in headless non-interactive environment.\n"
        "Please provide DEEPSEEK_TOKEN via environment variable or copy your session.json file."
    )
