"""SQLite storage (PRD "API dan data utama"). All times are stored as UTC ISO strings."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Jakarta',
    baseline TEXT,
    settings TEXT NOT NULL DEFAULT '{}',
    emergency_contacts TEXT NOT NULL DEFAULT '[]',
    calibration_started_at TEXT,
    is_simulated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('patient', 'caregiver')),
    nickname TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    patient_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sensor_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    timestamp TEXT,
    received_at TEXT,
    arrival_utc TEXT,
    quality_flag TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sensor_patient_arrival ON sensor_records (patient_id, arrival_utc);
CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    window_end TEXT NOT NULL,
    risk_level TEXT,
    reasons TEXT NOT NULL,
    model_version TEXT,
    data_quality TEXT,
    display_state TEXT NOT NULL,
    hr_mean REAL
);
CREATE INDEX IF NOT EXISTS ix_status_patient ON status_history (patient_id, id);
CREATE TABLE IF NOT EXISTS invites (
    code TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
CREATE TABLE IF NOT EXISTS circle_members (
    member_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    name TEXT,
    account_id TEXT,
    phone TEXT,
    web_token TEXT UNIQUE,
    joined_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS push_tokens (
    token TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    audience TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient TEXT,
    message TEXT NOT NULL,
    event_time TEXT NOT NULL,
    sent_at TEXT,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_notifications_patient ON notifications (patient_id, id);
CREATE TABLE IF NOT EXISTS therapist_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    highest_level TEXT NOT NULL,
    reasons TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {"share_reasons": True, "share_heart_rate": True, "notify_me_first": False}
INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
JSON_FIELDS = ("baseline", "settings", "emergency_contacts")


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _clean_json(value):
    """NaN is not JSON: baselines computed by pandas may contain it."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return value


def _dumps(value) -> str:
    return json.dumps(_clean_json(value), default=str)


class Database:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _one(self, sql: str, args=()) -> dict | None:
        row = self.conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # --- patients ------------------------------------------------------------------------

    @staticmethod
    def _patient(row: dict | None) -> dict | None:
        if row is None:
            return None
        for f in JSON_FIELDS:
            row[f] = json.loads(row[f]) if row[f] else None
        row["settings"] = {**DEFAULT_SETTINGS, **(row["settings"] or {})}
        row["emergency_contacts"] = row["emergency_contacts"] or []
        row["is_simulated"] = bool(row["is_simulated"])
        return row

    def get_patient(self, patient_id: str) -> dict | None:
        return self._patient(self._one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)))

    def ensure_patient(self, patient_id: str, nickname: str | None = None, timezone: str = "Asia/Jakarta",
                       baseline: dict | None = None, is_simulated: bool = False) -> dict:
        self.conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, nickname, timezone, baseline, is_simulated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (patient_id, nickname or patient_id, timezone, _dumps(baseline) if baseline else None,
             int(is_simulated), iso(utcnow())))
        return self.get_patient(patient_id)

    def update_patient(self, patient_id: str, **fields) -> dict:
        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            vals = [_dumps(v) if k in JSON_FIELDS and v is not None else v for k, v in fields.items()]
            self.conn.execute(f"UPDATE patients SET {cols} WHERE patient_id = ?", (*vals, patient_id))
        return self.get_patient(patient_id)

    def new_patient_id(self) -> str:
        while True:
            pid = "P-" + secrets.token_hex(4).upper()
            if not self.get_patient(pid):
                return pid

    # --- accounts ------------------------------------------------------------------------

    def create_account(self, role: str, nickname: str, patient_id: str | None) -> tuple[dict, str]:
        token = secrets.token_urlsafe(32)
        account_id = "A-" + secrets.token_hex(6)
        self.conn.execute(
            "INSERT INTO accounts (account_id, role, nickname, token_hash, patient_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (account_id, role, nickname, hash_token(token), patient_id, iso(utcnow())))
        return self.account(account_id), token

    def account(self, account_id: str) -> dict | None:
        row = self._one("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
        if row:
            row.pop("token_hash")
        return row

    def account_by_token(self, token: str) -> dict | None:
        row = self._one("SELECT * FROM accounts WHERE token_hash = ?", (hash_token(token),))
        if row:
            row.pop("token_hash")
        return row

    def patient_account(self, patient_id: str) -> dict | None:
        return self._one("SELECT account_id FROM accounts WHERE role = 'patient' AND patient_id = ?", (patient_id,))

    # --- sensor records ------------------------------------------------------------------

    def insert_records(self, rows: list[tuple]) -> None:
        """rows: (patient_id, timestamp, received_at, arrival_utc, quality_flag, payload dict)."""
        self.conn.executemany(
            "INSERT INTO sensor_records (patient_id, timestamp, received_at, arrival_utc, quality_flag, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(p, None if t is None else str(t), None if r is None else str(r), a, f, _dumps(d))
             for p, t, r, a, f, d in rows])

    def records_between(self, patient_id: str, start: str, end: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload FROM sensor_records WHERE patient_id = ? AND arrival_utc >= ? AND arrival_utc <= ? ORDER BY id",
            (patient_id, start, end)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # --- status --------------------------------------------------------------------------

    def add_status(self, status: dict, display_state: str, hr_mean: float | None) -> None:
        self.conn.execute(
            "INSERT INTO status_history (patient_id, window_end, risk_level, reasons, model_version, data_quality, "
            "display_state, hr_mean) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (status["patient_id"], status["window_end"], status["risk_level"], json.dumps(status["reasons"]),
             status["model_version"], status["data_quality"], display_state, hr_mean))

    def history(self, patient_id: str, limit: int = 180) -> list[dict]:
        rows = self._all("SELECT * FROM status_history WHERE patient_id = ? ORDER BY id DESC LIMIT ?", (patient_id, limit))
        for r in rows:
            r.pop("id")
            r["reasons"] = json.loads(r["reasons"])
        return rows[::-1]

    # --- circle --------------------------------------------------------------------------

    def create_invite(self, patient_id: str, ttl: pd.Timedelta) -> dict:
        now = utcnow()
        while True:
            code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(6))
            if not self._one("SELECT code FROM invites WHERE code = ?", (code,)):
                break
        self.conn.execute("INSERT INTO invites (code, patient_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                          (code, patient_id, iso(now), iso(now + ttl)))
        return self._one("SELECT * FROM invites WHERE code = ?", (code,))

    def use_invite(self, code: str) -> str | None:
        """Mark a valid, unused invite as used; returns its patient_id."""
        now = iso(utcnow())
        row = self._one("SELECT * FROM invites WHERE code = ? AND used_at IS NULL AND expires_at > ?",
                        (code.strip().upper(), now))
        if not row:
            return None
        self.conn.execute("UPDATE invites SET used_at = ? WHERE code = ?", (now, row["code"]))
        return row["patient_id"]

    def add_member(self, patient_id: str, name: str | None, account_id: str | None, phone: str | None,
                   web_token: str | None) -> dict:
        member_id = "M-" + secrets.token_hex(6)
        self.conn.execute(
            "INSERT INTO circle_members (member_id, patient_id, name, account_id, phone, web_token, joined_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (member_id, patient_id, name, account_id, phone, web_token, iso(utcnow())))
        return self.member(member_id)

    def member(self, member_id: str) -> dict | None:
        return self._one("SELECT * FROM circle_members WHERE member_id = ?", (member_id,))

    def members(self, patient_id: str) -> list[dict]:
        return self._all("SELECT * FROM circle_members WHERE patient_id = ? ORDER BY joined_at", (patient_id,))

    def active_member_by_web_token(self, token: str) -> dict | None:
        return self._one("SELECT * FROM circle_members WHERE web_token = ? AND revoked_at IS NULL", (token,))

    def active_membership(self, account_id: str) -> dict | None:
        return self._one("SELECT * FROM circle_members WHERE account_id = ? AND revoked_at IS NULL", (account_id,))

    def revoke_member(self, patient_id: str, member_id: str) -> dict | None:
        self.conn.execute("UPDATE circle_members SET revoked_at = ? WHERE patient_id = ? AND member_id = ? "
                          "AND revoked_at IS NULL", (iso(utcnow()), patient_id, member_id))
        return self.member(member_id)

    # --- push / whatsapp recipients ------------------------------------------------------

    def set_push_token(self, account_id: str, token: str) -> None:
        self.conn.execute("INSERT INTO push_tokens (token, account_id, updated_at) VALUES (?, ?, ?) "
                          "ON CONFLICT(token) DO UPDATE SET account_id = excluded.account_id, updated_at = excluded.updated_at",
                          (token, account_id, iso(utcnow())))

    def push_tokens(self, patient_id: str, audience: str) -> list[str]:
        tokens = []
        if audience in ("patient", "all"):
            tokens += [r["token"] for r in self._all(
                "SELECT t.token FROM push_tokens t JOIN accounts a ON a.account_id = t.account_id "
                "WHERE a.role = 'patient' AND a.patient_id = ?", (patient_id,))]
        if audience in ("caregivers", "all"):
            tokens += [r["token"] for r in self._all(
                "SELECT t.token FROM push_tokens t JOIN circle_members m ON m.account_id = t.account_id "
                "WHERE m.patient_id = ? AND m.revoked_at IS NULL", (patient_id,))]
        return tokens

    def whatsapp_recipients(self, patient_id: str) -> list[dict]:
        """Caregivers without the app: they have a phone number and a web link."""
        return self._all("SELECT member_id, phone, web_token FROM circle_members WHERE patient_id = ? "
                         "AND revoked_at IS NULL AND account_id IS NULL AND phone IS NOT NULL", (patient_id,))

    # --- notifications -------------------------------------------------------------------

    def add_notification(self, patient_id: str, kind: str, audience: str, channel: str, recipient: str | None,
                         message: str, event_time: str, status: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO notifications (patient_id, kind, audience, channel, recipient, message, event_time, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (patient_id, kind, audience, channel, recipient, message, event_time, status))
        return int(cur.lastrowid)

    def set_notification_status(self, ids: list[int], status: str, error: str | None = None) -> None:
        self.conn.executemany("UPDATE notifications SET status = ?, error = ?, sent_at = ? WHERE id = ?",
                              [(status, error, iso(utcnow()), i) for i in ids])

    def notifications(self, patient_id: str, role: str, limit: int = 50, channel: str | None = "in_app") -> list[dict]:
        """In-app notification feed as one role sees it (patient: patient/all, caregiver: caregivers/all)."""
        audiences = ("patient", "all") if role == "patient" else ("caregivers", "all")
        sql = "SELECT * FROM notifications WHERE patient_id = ? AND audience IN (?, ?)"
        args = [patient_id, *audiences]
        if channel:
            sql += " AND channel = ?"
            args.append(channel)
        rows = self._all(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))
        return rows[::-1]

    # --- therapist notes -----------------------------------------------------------------

    def add_note(self, patient_id: str, started_at: str, ended_at: str, highest_level: str, reasons: list[str],
                 summary: str) -> None:
        self.conn.execute(
            "INSERT INTO therapist_notes (patient_id, started_at, ended_at, highest_level, reasons, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (patient_id, started_at, ended_at, highest_level, json.dumps(reasons), summary, iso(utcnow())))

    def notes(self, patient_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM therapist_notes WHERE patient_id = ? ORDER BY id", (patient_id,))
        for r in rows:
            r["reasons"] = json.loads(r["reasons"])
        return rows
