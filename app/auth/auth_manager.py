"""
User authentication and thread persistence manager for PostgreSQL.
Provides secure password hashing (PBKDF2-HMAC-SHA256), session management,
and user-specific conversation thread tracking.
"""

import os
import hashlib
import secrets
import logging
from datetime import datetime
import psycopg
from app.config import get_settings
from app.memory.checkpointer import _clean_url

logger = logging.getLogger(__name__)


def _get_connection():
    """Get a direct connection to PostgreSQL for authentication queries."""
    settings = get_settings()
    conninfo = _clean_url(settings.postgres_url)
    return psycopg.connect(conninfo, autocommit=True)


def init_auth_db():
    """
    Initialize auth tables (`users` and `user_threads`) in PostgreSQL if they don't exist.
    """
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        full_name VARCHAR(150),
        password_hash VARCHAR(256) NOT NULL,
        salt VARCHAR(64) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS user_threads (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        thread_id VARCHAR(100) NOT NULL,
        name VARCHAR(200) NOT NULL,
        created_at VARCHAR(50) NOT NULL,
        last_updated VARCHAR(50) NOT NULL,
        UNIQUE (user_id, thread_id)
    );

    CREATE INDEX IF NOT EXISTS idx_user_threads_user_id ON user_threads(user_id);
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_tables_sql)
        logger.info("[Auth] Database tables verified / created.")
    except Exception as e:
        logger.error(f"[Auth] Error initializing database tables: {e}")
        raise e


def _hash_password(password: str, salt: str) -> str:
    """Hash password with PBKDF2-HMAC-SHA256 (200,000 rounds)."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200000,
    ).hex()


def signup_user(username: str, password: str, full_name: str = "") -> tuple[bool, str]:
    """
    Register a new user account with salted password hash.
    
    Returns:
        (success: bool, message: str)
    """
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    return False, f"Username '{username}' is already taken. Please choose another or sign in."

                cur.execute(
                    """
                    INSERT INTO users (username, full_name, password_hash, salt, created_at, last_login)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (username, full_name.strip() or username.capitalize(), password_hash, salt),
                )
        return True, "Account created successfully! You can now log in."
    except Exception as e:
        logger.error(f"[Auth] Sign up error: {e}")
        return False, f"Registration failed: {str(e)}"


def authenticate_user(username: str, password: str) -> tuple[dict | None, str]:
    """
    Authenticate a user by username and password.

    Returns:
        (user_dict | None, message: str)
    """
    username = username.strip().lower()
    if not username or not password:
        return None, "Please enter both username and password."

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, full_name, password_hash, salt, created_at
                    FROM users WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
                if not row:
                    return None, "Invalid username or password."

                user_id, uname, full_name, expected_hash, salt, created_at = row
                candidate_hash = _hash_password(password, salt)

                if secrets.compare_digest(candidate_hash, expected_hash):
                    # Update last login timestamp
                    cur.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                        (user_id,),
                    )
                    user_data = {
                        "id": user_id,
                        "username": uname,
                        "full_name": full_name,
                        "created_at": created_at.strftime("%b %d, %Y") if created_at else "",
                    }
                    return user_data, "Login successful!"
                else:
                    return None, "Invalid username or password."
    except Exception as e:
        logger.error(f"[Auth] Login error: {e}")
        return None, f"Login failed: {str(e)}"


def get_user_threads(user_id: int) -> list[dict]:
    """
    Retrieve all conversation threads for a specific user from PostgreSQL.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT thread_id, name, created_at, last_updated
                    FROM user_threads
                    WHERE user_id = %s
                    ORDER BY id ASC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "thread_id": r[0],
                        "name": r[1],
                        "created_at": r[2],
                        "last_updated": r[3],
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.error(f"[Auth] Error fetching user threads: {e}")
        return []


def save_user_thread(user_id: int, thread_id: str, name: str, created_at: str, last_updated: str):
    """
    Upsert user thread record in PostgreSQL.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_threads (user_id, thread_id, name, created_at, last_updated)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, thread_id)
                    DO UPDATE SET name = EXCLUDED.name, last_updated = EXCLUDED.last_updated
                    """,
                    (user_id, thread_id, name, created_at, last_updated),
                )
    except Exception as e:
        logger.error(f"[Auth] Error saving user thread: {e}")


def delete_user_thread(user_id: int, thread_id: str):
    """
    Delete a user thread record.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_threads WHERE user_id = %s AND thread_id = %s",
                    (user_id, thread_id),
                )
    except Exception as e:
        logger.error(f"[Auth] Error deleting user thread: {e}")
