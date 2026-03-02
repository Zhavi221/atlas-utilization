"""
SQLite-backed shard storage for invariant-mass arrays.

This module stores arrays in a small number of files to avoid inode/file-count
explosion from per-combination .npy outputs.
"""

from __future__ import annotations

import io
import os
import sqlite3
import zlib
from typing import Dict, Iterator, List, Optional

import numpy as np


class SqliteArrayShardWriter:
    """Append-only writer for array shards keyed by signature."""

    def __init__(self, db_path: str, table_name: str = "array_chunks"):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT NOT NULL,
                n_entries INTEGER NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signature ON {table_name}(signature)"
        )

    def append_array(self, signature: str, arr: np.ndarray) -> None:
        """Append one numpy array chunk under a signature."""
        if arr.size == 0:
            return
        payload = _serialize_array(arr)
        self.conn.execute(
            f"INSERT INTO {self.table_name}(signature, n_entries, payload) VALUES (?, ?, ?)",
            (signature, int(arr.size), payload),
        )

    def append_many(self, signature_to_array: Dict[str, np.ndarray]) -> int:
        """Append many arrays in a single transaction."""
        rows = []
        for signature, arr in signature_to_array.items():
            if arr.size == 0:
                continue
            rows.append((signature, int(arr.size), _serialize_array(arr)))
        if not rows:
            return 0
        self.conn.executemany(
            f"INSERT INTO {self.table_name}(signature, n_entries, payload) VALUES (?, ?, ?)",
            rows,
        )
        return len(rows)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def list_signatures(db_path: str, table_name: str = "array_chunks") -> List[str]:
    """Return all distinct signatures in a shard DB."""
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT signature FROM {table_name} ORDER BY signature"
        ).fetchall()
    return [r[0] for r in rows]


def iter_arrays_for_signature(
    db_path: str,
    signature: str,
    table_name: str = "array_chunks",
) -> Iterator[np.ndarray]:
    """Yield all chunks for one signature."""
    if not os.path.exists(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT payload FROM {table_name} WHERE signature = ?",
            (signature,),
        ).fetchall()
    for (payload,) in rows:
        yield _deserialize_array(payload)


def iter_all_chunks(
    db_path: str, table_name: str = "array_chunks"
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (signature, chunk_array) for every row."""
    if not os.path.exists(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT signature, payload FROM {table_name} ORDER BY id"
        ).fetchall()
    for signature, payload in rows:
        yield signature, _deserialize_array(payload)


def get_total_entries(db_path: str, table_name: str = "array_chunks") -> int:
    """Return total entry count from metadata column."""
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COALESCE(SUM(n_entries), 0) FROM {table_name}").fetchone()
    return int(row[0] if row else 0)


def _serialize_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=1)


def _deserialize_array(payload: bytes) -> np.ndarray:
    raw = zlib.decompress(payload)
    buffer = io.BytesIO(raw)
    return np.load(buffer, allow_pickle=False)
