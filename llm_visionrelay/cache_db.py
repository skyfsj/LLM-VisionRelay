"""SQLite-backed persistent cache with WAL mode.

Stores tenant-isolated metadata: image registry, URL aliases, vision summary
cache, and targeted query cache. Object bytes live in the content-addressed
file store (see :mod:`llm_visionrelay.image_store`).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from llm_visionrelay.errors import CacheDatabaseError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    tenant_id      TEXT NOT NULL,
    image_sha256   TEXT NOT NULL,
    mime_type      TEXT,
    width          INTEGER,
    height         INTEGER,
    size_bytes     INTEGER NOT NULL,
    source_kind    TEXT,
    created_at     REAL NOT NULL,
    PRIMARY KEY (tenant_id, image_sha256)
);

CREATE TABLE IF NOT EXISTS url_aliases (
    tenant_id      TEXT NOT NULL,
    url_hash       TEXT NOT NULL,
    url            TEXT NOT NULL,
    image_sha256   TEXT NOT NULL,
    etag           TEXT,
    last_modified  TEXT,
    expires_at     REAL NOT NULL,
    created_at     REAL NOT NULL,
    PRIMARY KEY (tenant_id, url_hash)
);

CREATE TABLE IF NOT EXISTS vision_summaries (
    tenant_id        TEXT NOT NULL,
    image_sha256     TEXT NOT NULL,
    vision_base_hash TEXT NOT NULL,
    vision_model     TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    schema_version   TEXT NOT NULL,
    detail_mode      TEXT NOT NULL,
    params_hash      TEXT NOT NULL DEFAULT '',
    result_json      TEXT NOT NULL,
    created_at       REAL NOT NULL,
    expires_at       REAL NOT NULL,
    PRIMARY KEY (tenant_id, image_sha256, vision_base_hash, vision_model,
                 prompt_version, schema_version, detail_mode, params_hash)
);

CREATE TABLE IF NOT EXISTS vision_queries (
    tenant_id        TEXT NOT NULL,
    image_sha256     TEXT NOT NULL,
    vision_base_hash TEXT NOT NULL,
    vision_model     TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    schema_version   TEXT NOT NULL,
    mode             TEXT NOT NULL,
    query_hash       TEXT NOT NULL,
    bbox_json        TEXT NOT NULL,
    tool_version     TEXT NOT NULL,
    params_hash      TEXT NOT NULL DEFAULT '',
    result_json      TEXT NOT NULL,
    created_at       REAL NOT NULL,
    expires_at       REAL NOT NULL,
    PRIMARY KEY (tenant_id, image_sha256, vision_base_hash, vision_model,
                 prompt_version, schema_version, mode, query_hash, bbox_json,
                 tool_version, params_hash)
);

CREATE INDEX IF NOT EXISTS idx_images_tenant ON images(tenant_id);
CREATE INDEX IF NOT EXISTS idx_url_aliases_expires ON url_aliases(expires_at);
CREATE INDEX IF NOT EXISTS idx_summaries_expires ON vision_summaries(expires_at);
CREATE INDEX IF NOT EXISTS idx_queries_expires ON vision_queries(expires_at);
"""


class CacheDB:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(str(self._path))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.executescript(_SCHEMA)
            await self._ensure_column("vision_summaries", "params_hash", "TEXT NOT NULL DEFAULT ''")
            await self._ensure_column("vision_queries", "params_hash", "TEXT NOT NULL DEFAULT ''")
            await self._conn.commit()
        except Exception as exc:  # pragma: no cover - defensive
            raise CacheDatabaseError(str(exc)) from exc

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    aclose = close

    async def _ensure_column(self, table: str, column: str, definition: str) -> None:
        try:
            cur = await self._conn.execute(f"PRAGMA table_info({table})")
            rows = await cur.fetchall()
            if column not in {r[1] for r in rows}:
                await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:  # pragma: no cover - defensive
            pass

    @property
    def path(self) -> Path:
        return self._path

    @asynccontextmanager
    async def transaction(self):
        if self._conn is None:
            raise CacheDatabaseError("cache database not connected")
        try:
            await self._conn.execute("BEGIN")
            yield
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    def _check(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise CacheDatabaseError("cache database not connected")
        return self._conn

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        try:
            conn = self._check()
            await conn.execute(sql, params)
            await conn.commit()
        except Exception as exc:
            raise CacheDatabaseError(str(exc)) from exc

    async def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        try:
            conn = self._check()
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            raise CacheDatabaseError(str(exc)) from exc

    async def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = await self._fetchall(sql, params)
        return rows[0] if rows else None

    # --- image registry -------------------------------------------------
    async def register_image(
        self,
        tenant: str,
        sha: str,
        mime: str | None,
        width: int | None,
        height: int | None,
        size_bytes: int,
        source_kind: str,
    ) -> None:
        await self._execute(
            """INSERT OR REPLACE INTO images
               (tenant_id, image_sha256, mime_type, width, height, size_bytes, source_kind, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tenant, sha, mime, width, height, size_bytes, source_kind, time.time()),
        )

    async def get_registered_image(self, tenant: str, sha: str) -> dict[str, Any] | None:
        return await self._fetchone(
            "SELECT * FROM images WHERE tenant_id = ? AND image_sha256 = ?",
            (tenant, sha),
        )

    # --- URL aliases ----------------------------------------------------
    async def get_url_alias(self, tenant: str, url_hash: str) -> dict[str, Any] | None:
        return await self._fetchone(
            "SELECT * FROM url_aliases WHERE tenant_id = ? AND url_hash = ?",
            (tenant, url_hash),
        )

    async def put_url_alias(
        self,
        tenant: str,
        url_hash: str,
        url: str,
        sha: str,
        etag: str | None,
        last_modified: str | None,
        expires_at: float,
    ) -> None:
        await self._execute(
            """INSERT OR REPLACE INTO url_aliases
               (tenant_id, url_hash, url, image_sha256, etag, last_modified, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tenant, url_hash, url, sha, etag, last_modified, expires_at, time.time()),
        )

    async def touch_url_alias(self, tenant: str, url_hash: str, expires_at: float) -> None:
        await self._execute(
            "UPDATE url_aliases SET expires_at = ? WHERE tenant_id = ? AND url_hash = ?",
            (expires_at, tenant, url_hash),
        )

    # --- vision summaries -----------------------------------------------
    async def get_summary(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        detail: str,
        params_hash: str,
        now: float,
    ) -> dict[str, Any] | None:
        return await self._fetchone(
            """SELECT * FROM vision_summaries
               WHERE tenant_id = ? AND image_sha256 = ? AND vision_base_hash = ?
                 AND vision_model = ? AND prompt_version = ? AND schema_version = ?
                 AND detail_mode = ? AND params_hash = ? AND expires_at > ?""",
            (tenant, sha, base_hash, model, prompt_ver, schema_ver, detail, params_hash, now),
        )

    async def get_any_summary(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        detail: str,
        params_hash: str,
    ) -> dict[str, Any] | None:
        return await self._fetchone(
            """SELECT * FROM vision_summaries
               WHERE tenant_id = ? AND image_sha256 = ? AND vision_base_hash = ?
                 AND vision_model = ? AND prompt_version = ? AND schema_version = ?
                 AND detail_mode = ? AND params_hash = ?""",
            (tenant, sha, base_hash, model, prompt_ver, schema_ver, detail, params_hash),
        )

    async def put_summary(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        detail: str,
        params_hash: str,
        result_json: str,
        created_at: float,
        expires_at: float,
    ) -> None:
        await self._execute(
            """INSERT OR REPLACE INTO vision_summaries
               (tenant_id, image_sha256, vision_base_hash, vision_model,
                prompt_version, schema_version, detail_mode, params_hash,
                result_json, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tenant,
                sha,
                base_hash,
                model,
                prompt_ver,
                schema_ver,
                detail,
                params_hash,
                result_json,
                created_at,
                expires_at,
            ),
        )

    # --- vision queries -------------------------------------------------
    async def get_query(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        mode: str,
        query_hash: str,
        bbox_json: str,
        tool_ver: str,
        params_hash: str,
        now: float,
    ) -> dict[str, Any] | None:
        return await self._fetchone(
            """SELECT * FROM vision_queries
               WHERE tenant_id = ? AND image_sha256 = ? AND vision_base_hash = ?
                 AND vision_model = ? AND prompt_version = ? AND schema_version = ?
                 AND mode = ? AND query_hash = ? AND bbox_json = ? AND tool_version = ?
                 AND params_hash = ? AND expires_at > ?""",
            (
                tenant,
                sha,
                base_hash,
                model,
                prompt_ver,
                schema_ver,
                mode,
                query_hash,
                bbox_json,
                tool_ver,
                params_hash,
                now,
            ),
        )

    async def get_any_query(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        mode: str,
        query_hash: str,
        bbox_json: str,
        tool_ver: str,
        params_hash: str,
    ) -> dict[str, Any] | None:
        return await self._fetchone(
            """SELECT * FROM vision_queries
               WHERE tenant_id = ? AND image_sha256 = ? AND vision_base_hash = ?
                 AND vision_model = ? AND prompt_version = ? AND schema_version = ?
                 AND mode = ? AND query_hash = ? AND bbox_json = ? AND tool_version = ?
                 AND params_hash = ?""",
            (
                tenant,
                sha,
                base_hash,
                model,
                prompt_ver,
                schema_ver,
                mode,
                query_hash,
                bbox_json,
                tool_ver,
                params_hash,
            ),
        )

    async def put_query(
        self,
        tenant: str,
        sha: str,
        base_hash: str,
        model: str,
        prompt_ver: str,
        schema_ver: str,
        mode: str,
        query_hash: str,
        bbox_json: str,
        tool_ver: str,
        params_hash: str,
        result_json: str,
        created_at: float,
        expires_at: float,
    ) -> None:
        await self._execute(
            """INSERT OR REPLACE INTO vision_queries
               (tenant_id, image_sha256, vision_base_hash, vision_model,
                prompt_version, schema_version, mode, query_hash, bbox_json,
                tool_version, params_hash, result_json, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tenant,
                sha,
                base_hash,
                model,
                prompt_ver,
                schema_ver,
                mode,
                query_hash,
                bbox_json,
                tool_ver,
                params_hash,
                result_json,
                created_at,
                expires_at,
            ),
        )

    # --- stats / purge / gc ---------------------------------------------
    async def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("images", "url_aliases", "vision_summaries", "vision_queries"):
            row = await self._fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(row["n"]) if row else 0
        return out

    async def referenced_objects(self) -> set[str]:
        rows = await self._fetchall(
            """SELECT image_sha256 AS s FROM images
               UNION SELECT image_sha256 FROM url_aliases
               UNION SELECT image_sha256 FROM vision_summaries
               UNION SELECT image_sha256 FROM vision_queries"""
        )
        return {r["s"] for r in rows}

    async def purge_expired(self, now: float) -> int:
        total = 0
        async with self.transaction():
            for table in ("vision_summaries", "vision_queries"):
                cur = await self._check().execute(f"DELETE FROM {table} WHERE expires_at < ?", (now,))
                total += cur.rowcount
            cur = await self._check().execute("DELETE FROM url_aliases WHERE expires_at < ?", (now,))
            total += cur.rowcount
        return total

    async def purge_all(self) -> None:
        async with self.transaction():
            for table in ("images", "url_aliases", "vision_summaries", "vision_queries"):
                await self._check().execute(f"DELETE FROM {table}")

    async def purge_tenant(self, tenant: str) -> None:
        async with self.transaction():
            for table in ("images", "url_aliases", "vision_summaries", "vision_queries"):
                await self._check().execute(f"DELETE FROM {table} WHERE tenant_id = ?", (tenant,))

    async def purge_sha(self, sha: str) -> None:
        async with self.transaction():
            for table in ("images", "url_aliases", "vision_summaries", "vision_queries"):
                await self._check().execute(f"DELETE FROM {table} WHERE image_sha256 = ?", (sha,))
