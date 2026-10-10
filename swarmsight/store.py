"""SQLite persistence: road events, observations (with provenance), sources, charger observations."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'phone',
    alpha REAL NOT NULL DEFAULT 2.0,
    beta REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS road_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    lat REAL, lon REAL,
    radius_m REAL NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    expires_at REAL NOT NULL,
    n_obs INTEGER NOT NULL DEFAULT 0,
    n_sources INTEGER NOT NULL DEFAULT 0,
    evidence TEXT,
    history TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES road_events(id),
    source_id TEXT NOT NULL,
    type TEXT NOT NULL,
    lat REAL, lon REAL,
    gps_accuracy_m REAL,
    conf REAL NOT NULL,
    severity REAL NOT NULL,
    observed_at REAL NOT NULL,
    received_at REAL NOT NULL,
    model_version TEXT,
    media TEXT,
    bbox TEXT,
    extra TEXT
);
CREATE TABLE IF NOT EXISTS charger_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site TEXT NOT NULL,
    bay TEXT NOT NULL,
    state TEXT NOT NULL,
    conf REAL,
    vehicle_cls TEXT,
    observed_at REAL NOT NULL,
    source_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_type ON road_events(type, status);
CREATE INDEX IF NOT EXISTS idx_obs_event ON observations(event_id);
"""


class Store:
    def __init__(self, path: str | Path | None = None):
        path = path or os.environ.get("SWARMSIGHT_DB", "runtime/swarmsight.db")
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def execute(self, sql: str, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def query(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # --- sources ----------------------------------------------------------------------
    def source(self, source_id: str, kind: str = "phone") -> dict:
        rows = self.query("SELECT * FROM sources WHERE source_id=?", (source_id,))
        if rows:
            return rows[0]
        self.execute("INSERT INTO sources(source_id, kind, created_at) VALUES (?,?,?)", (source_id, kind, time.time()))
        return self.query("SELECT * FROM sources WHERE source_id=?", (source_id,))[0]

    def update_source(self, source_id: str, d_alpha: float = 0.0, d_beta: float = 0.0):
        self.source(source_id)
        self.execute("UPDATE sources SET alpha=alpha+?, beta=beta+? WHERE source_id=?", (d_alpha, d_beta, source_id))

    # --- events -----------------------------------------------------------------------
    def event(self, eid: int) -> dict | None:
        rows = self.query("SELECT * FROM road_events WHERE id=?", (eid,))
        if not rows:
            return None
        e = rows[0]
        e["history"] = json.loads(e["history"] or "[]")
        return e

    def events(self, where: str = "1=1", args=()) -> list[dict]:
        rows = self.query(f"SELECT * FROM road_events WHERE {where} ORDER BY last_seen DESC", args)
        for e in rows:
            e["history"] = json.loads(e["history"] or "[]")
        return rows

    def observations_for(self, eid: int) -> list[dict]:
        return self.query("SELECT * FROM observations WHERE event_id=? ORDER BY observed_at", (eid,))
