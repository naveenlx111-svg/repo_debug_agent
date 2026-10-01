"""On-disk cache of LLM replies, keyed by the exact request.

Used for file reviews: re-running the agent re-reviews only files whose review
prompt changed (different code, hints, test output, prompt wording or model).
The raw reply is stored, so improvements to reply parsing still apply to it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class ReplyCache:
    def __init__(self, root: Path):
        self.root = root
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, messages: list[dict[str, str]], **params: Any) -> str:
        blob = json.dumps([model, messages, params], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> tuple[str, bool] | None:
        """(reply, truncated) or None. A truncated entry holds the partial reply, so a
        re-run can skip a request that is known to run into the token limit."""
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            found = (str(entry["reply"]), bool(entry.get("truncated", False)))
        except (OSError, ValueError, KeyError, TypeError):
            found = None
        with self._lock:
            if found is None:
                self.misses += 1
            else:
                self.hits += 1
        return found

    def put(self, key: str, reply: str, truncated: bool = False) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so concurrent runs never see a half-written entry.
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"reply": reply, "truncated": truncated}, f)
            os.replace(tmp, path)
        except OSError:
            pass  # a cache that can't be written is just a slower run
