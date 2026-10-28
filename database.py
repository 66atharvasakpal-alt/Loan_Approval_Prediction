"""
database.py
-----------
SQLite storage for registered users and prediction history.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "loan_app.db"

_ITERATIONS = 200_000


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _ITERATIONS
    ).hex()


def init_db() -> None:
    """Create tables. If an old-format database is found, back it up first."""
    if DB_PATH.exists():
        with closing(_connect()) as conn:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)")]
        if cols and "password_hash" not in cols:
            DB_PATH.rename(DB_PATH.with_suffix(".db.bak"))

    with closing(_connect()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                applicant_name TEXT,
                inputs_json    TEXT NOT NULL,
                prediction     TEXT NOT NULL,
                probability    REAL NOT NULL,
                model_name     TEXT NOT NULL,
                created_at     TEXT NOT NULL
            );
            """
        )


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return False, "Username must be 3-30 characters: letters, numbers or underscore."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    salt = os.urandom(16)
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    username,
                    _hash_password(password, salt),
                    salt.hex(),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def authenticate(username: str, password: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    candidate = _hash_password(password, bytes.fromhex(row["salt"]))
    if hmac.compare_digest(candidate, row["password_hash"]):
        return {"id": row["id"], "username": row["username"]}
    return None


def save_prediction(
    user_id: int,
    applicant_name: str,
    inputs: dict,
    prediction: str,
    probability: float,
    model_name: str,
) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions "
            "(user_id, applicant_name, inputs_json, prediction, probability, "
            "model_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                applicant_name.strip() or None,
                json.dumps(inputs),
                prediction,
                float(probability),
                model_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def get_history(user_id: int) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history(user_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
