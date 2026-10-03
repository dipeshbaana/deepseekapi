"""
Server package providing OpenAI-compatible endpoints for DeepSeek Web Client.
"""

from .app import app, run_server

__all__ = ["app", "run_server"]
