"""listing_record (optional): the CoStar residential listing behind a scan.

NOT VERIFIED against the real kit: no kit link or password was available when
this was written. Staff described the kit as a zip of JSONL with field docs
inside; the Matterport quickstart reads a `residential-listings.csv` from it.
So this code looks at what is actually in the zip and handles both, and
`kit_info()` prints the file list and columns to settle the question.

The kit is "hackathon use only". Never commit it or anything read from it.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile

import requests

from .. import config


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, so one member of a
    remote zip can be read without downloading all 160 MB."""

    def __init__(self, url: str, block: int = 4 * 2 ** 20):
        self.url, self.pos, self.block, self.buf, self.buf_at = url, 0, block, b"", -1
        head = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.buf_at <= self.pos and self.pos + n <= self.buf_at + len(self.buf)):
            end = min(self.size, self.pos + max(n, self.block)) - 1
            resp = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
            resp.raise_for_status()
            self.buf_at, self.buf = self.pos, resp.content
        out = self.buf[self.pos - self.buf_at: self.pos - self.buf_at + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_kit() -> zipfile.ZipFile | None:
    """The kit zip from COSTAR_KIT_ZIP (local path) or COSTAR_KIT_URL. None when not configured."""
    path, url = config.env("COSTAR_KIT_ZIP"), config.env("COSTAR_KIT_URL")
    password = config.env("COSTAR_ZIP_PASSWORD")
    if not password or not (path or url):
        return None
    kit = zipfile.ZipFile(path if path else HttpFile(url))
    kit.setpassword(password.encode())
    return kit


def _residential_members(kit: zipfile.ZipFile) -> list[str]:
    names = [n for n in kit.namelist() if n.lower().endswith((".csv", ".jsonl", ".ndjson"))]
    return [n for n in names if "residential" in n.lower() or "listing" in n.lower() or "homes" in n.lower()]


def _rows(kit: zipfile.ZipFile, member: str):
    with kit.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        if member.lower().endswith(".csv"):
            yield from csv.DictReader(text)
        else:
            for line in text:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def kit_info() -> dict:
    """What is in the kit: every file name, and the columns of the listing files."""
    kit = open_kit()
    if kit is None:
        return {"configured": False, "reason": "Set COSTAR_ZIP_PASSWORD and COSTAR_KIT_ZIP or COSTAR_KIT_URL."}
    info = {"configured": True, "files": kit.namelist()[:200], "listing_files": {}}
    for member in _residential_members(kit):
        first = next(_rows(kit, member), None)
        info["listing_files"][member] = sorted(first) if isinstance(first, dict) else None
    return info


def listing_record(homes_listing_id: str) -> dict:
    """The residential listing row whose id equals homes_listing_id, or found=False."""
    kit = open_kit()
    if kit is None:
        return {"found": False, "reason": "CoStar kit not configured."}
    wanted = str(homes_listing_id).strip()
    try:
        for member in _residential_members(kit):
            for row in _rows(kit, member):
                if isinstance(row, dict) and any(str(v).strip() == wanted for v in row.values()
                                                 if isinstance(v, (str, int))):
                    return {"found": True, "file": member, "listing": row}
    except RuntimeError as exc:         # zipfile raises RuntimeError on a wrong password
        return {"found": False, "reason": f"Could not read the kit: {exc}. Type the zip password by hand."}
    return {"found": False, "reason": "No listing with this id in the kit."}
