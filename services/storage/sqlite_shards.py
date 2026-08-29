"""
SQLite-backed shard storage for invariant-mass arrays.

This module stores arrays in a small number of files to avoid inode/file-count
explosion from per-combination .npy outputs.
"""

from __future__ import annotations

import io
import os
import re
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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS final_state_counts (
                final_state TEXT NOT NULL,
                n_events INTEGER NOT NULL
            )
            """
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

    def record_final_state_count(self, final_state: str, n_events: int) -> None:
        """Record population before combination-specific physics cuts."""
        self.conn.execute(
            "INSERT INTO final_state_counts(final_state, n_events) VALUES (?, ?)",
            (final_state, int(n_events)),
        )

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


def prune_final_states_below_min_events(
    db_path: str | list[str],
    min_events: int,
    table_name: str = "array_chunks",
) -> list[str]:
    """Remove final states whose global event population is below a threshold.

    Chunk/file prefixes are ignored.  For each final state, the largest global
    entry count among its invariant-mass combinations is its event population:
    every eligible event contributes once to at least one contained combination.
    This uses the uncompressed SQLite metadata and does not reread ROOT payloads.
    """
    db_paths = [db_path] if isinstance(db_path, str) else list(db_path)
    db_paths = [path for path in db_paths if os.path.exists(path)]
    if min_events <= 1 or not db_paths:
        return []

    pattern = re.compile(r"(_FS_[0-9a-z_]+)_IM_([0-9a-z]+)$")
    totals_by_channel: dict[tuple[str, str], int] = {}
    explicit_populations: dict[str, int] = {}
    signatures_by_db_and_fs: dict[tuple[str, str], list[str]] = {}
    for path in db_paths:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                f"""
                SELECT signature, COALESCE(SUM(n_entries), 0)
                FROM {table_name}
                GROUP BY signature
                """
            ).fetchall()
            has_count_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='final_state_counts'"
            ).fetchone()
            if has_count_table:
                count_rows = conn.execute(
                    """
                    SELECT final_state, COALESCE(SUM(n_events), 0)
                    FROM final_state_counts
                    GROUP BY final_state
                    """
                ).fetchall()
                for final_state, count in count_rows:
                    normalized = (
                        final_state if final_state.startswith("_FS_")
                        else f"_FS_{final_state}"
                    )
                    explicit_populations[normalized] = (
                        explicit_populations.get(normalized, 0) + int(count)
                    )
        for signature, entries in rows:
            match = pattern.search(signature)
            if not match:
                continue
            final_state, combination = match.groups()
            key = (final_state, combination)
            totals_by_channel[key] = totals_by_channel.get(key, 0) + int(entries)
            signatures_by_db_and_fs.setdefault((path, final_state), []).append(signature)

    populations: dict[str, int] = {}
    for (final_state, _combination), entries in totals_by_channel.items():
        populations[final_state] = max(populations.get(final_state, 0), entries)
    populations.update(explicit_populations)

    removed = [fs for fs, count in populations.items() if count < min_events]
    for path in db_paths:
        with sqlite3.connect(path) as conn:
            signatures = []
            for final_state in removed:
                signatures.extend(signatures_by_db_and_fs.get((path, final_state), []))
            conn.executemany(
                f"DELETE FROM {table_name} WHERE signature = ?",
                [(signature,) for signature in signatures],
            )
            conn.commit()
    return sorted(removed)


def _serialize_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=1)


def _deserialize_array(payload: bytes) -> np.ndarray:
    raw = zlib.decompress(payload)
    buffer = io.BytesIO(raw)
    return np.load(buffer, allow_pickle=False)
