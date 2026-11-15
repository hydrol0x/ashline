"""Settings read from the environment and a git-ignored .env file.

Keys are never printed or written anywhere by this package.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INDEX_URL = (
    "https://static.matterport.com/hackathon/26.15.39-main-39-gef8d668/"
    "assets/quickstart/models_index.csv"
)


class MissingKey(RuntimeError):
    """A required key is not set in the environment or in .env."""


def _dotenv_paths() -> list[Path]:
    return [Path.cwd() / ".env", REPO_ROOT / ".env"]


def load_env() -> None:
    """Load KEY=VALUE lines from .env without overriding the real environment."""
    for path in _dotenv_paths():
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if value[:1] in ("'", '"') and value[:1] in value[1:]:
                value = value[1:value.index(value[0], 1)]        # quoted: keep everything inside
            else:
                value = "" if value.startswith("#") else value.split(" #", 1)[0].strip()   # " #" starts a comment
            if value:
                os.environ.setdefault(key.strip(), value)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    load_env()
    value = os.environ.get(name) or default
    if required and not value:
        raise MissingKey(f"{name} is not set. Put it in .env (see .env.example).")
    return value


def data_dir() -> Path:
    """Root of all case data. Git-ignored: it holds scan images and addresses."""
    root = Path(env("ASHLINE_DATA_DIR", "cases"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def case_dir(case_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)
    path = data_dir() / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = REPO_ROOT / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
