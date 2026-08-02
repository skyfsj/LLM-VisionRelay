"""Content-addressed object storage on the local filesystem.

Objects are stored at ``{cache_dir}/objects/sha256/ab/cd/<full-sha256>`` using
atomic writes (temp file + ``os.replace``). Lookups are derived from a validated
64-hex digest only, so user-controlled URLs are never concatenated into paths.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from llm_visionrelay.errors import InvalidImageRef
from llm_visionrelay.security import is_hex64


class ImageStore:
    def __init__(self, cache_dir: Path, gc_object_min_age: float = 3600.0) -> None:
        self._root = Path(cache_dir)
        self._objects = self._root / "objects" / "sha256"
        self._gc_object_min_age = gc_object_min_age
        self._objects.mkdir(parents=True, exist_ok=True)

    def object_path(self, sha: str) -> Path:
        if not is_hex64(sha):
            raise InvalidImageRef("invalid image digest")
        return self._objects / sha[:2] / sha[2:4] / sha

    def store_bytes(self, data: bytes) -> str:
        import hashlib

        sha = hashlib.sha256(data).hexdigest()
        dest = self.object_path(sha)
        if dest.exists() and dest.stat().st_size == len(data):
            return sha
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".tmp-{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return sha

    def read_object(self, sha: str) -> bytes | None:
        path = self.object_path(sha)
        try:
            return path.read_bytes()
        except (OSError, ValueError):
            return None

    def object_exists(self, sha: str) -> bool:
        try:
            return self.object_path(sha).is_file()
        except InvalidImageRef:
            return False

    def delete_object(self, sha: str) -> None:
        try:
            path = self.object_path(sha)
            if path.exists():
                path.unlink(missing_ok=True)
                self._prune_empty_dirs(path.parent)
        except InvalidImageRef:
            pass

    def delete_all_objects(self) -> None:
        if self._objects.exists():
            for root, _dirs, files in os.walk(self._objects, topdown=False):
                for name in files:
                    (Path(root) / name).unlink(missing_ok=True)

    def iterate_object_shas(self) -> list[str]:
        shas: list[str] = []
        if not self._objects.exists():
            return shas
        for root, _dirs, files in os.walk(self._objects):
            for name in files:
                if is_hex64(name):
                    shas.append(name)
        return shas

    def iterate_old_object_shas(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        if not self._objects.exists():
            return out
        cutoff = time.time() - self._gc_object_min_age
        for root, _dirs, files in os.walk(self._objects):
            for name in files:
                if not is_hex64(name):
                    continue
                p = Path(root) / name
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    out.append((name, mtime))
        return out

    def cache_dir_size_bytes(self) -> int:
        total = 0
        if not self._objects.exists():
            return 0
        for root, _dirs, files in os.walk(self._objects):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
        return total

    @staticmethod
    def _prune_empty_dirs(path: Path) -> None:
        while path != path.anchor:
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent
