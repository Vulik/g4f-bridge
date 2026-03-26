"""
Persistent storage module using SQLite.
No top-level optional imports — graceful fallback.
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from logger_setup import get_logger


class Storage:
    """SQLite-based persistent storage with connection pooling."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        from environment import get_environment

        self.logger = get_logger("storage")
        env = get_environment()

        if db_path is None:
            db_path = env.data_dir / "bridge_data.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()

        # Initialize tables
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection (reuse per thread)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=10,
                check_same_thread=False,
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                client_id TEXT DEFAULT 'picoclaw',
                messages TEXT,
                token_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS provider_stats (
                provider_name TEXT,
                model TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                total_response_time REAL DEFAULT 0.0,
                last_used TEXT,
                last_error TEXT,
                status TEXT DEFAULT 'untested',
                PRIMARY KEY (provider_name, model)
            );

            CREATE TABLE IF NOT EXISTS config_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS update_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_version TEXT,
                to_version TEXT,
                status TEXT,
                timestamp TEXT DEFAULT (datetime('now')),
                error_log TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_conv_updated
                ON conversations(updated_at);
            CREATE INDEX IF NOT EXISTS idx_stats_status
                ON provider_stats(status);
        """)

        conn.commit()
        self.logger.info(f"Database initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        token_count: int,
        client_id: str = "picoclaw",
    ) -> None:
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO conversations
                   (id, client_id, messages, token_count, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (conversation_id, client_id, json.dumps(messages), token_count),
            )
            conn.commit()

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, client_id, messages, token_count, created_at, updated_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if row:
            return {
                "id": row[0],
                "client_id": row[1],
                "messages": json.loads(row[2]),
                "token_count": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
        return None

    def cleanup_old_conversations(self, hours: int = 168) -> int:
        conn = self._get_conn()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE updated_at < datetime('now', ?)",
                (f"-{hours} hours",),
            )
            conn.commit()
            deleted = cursor.rowcount
        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} old conversations")
        return deleted

    # ------------------------------------------------------------------
    # Provider stats
    # ------------------------------------------------------------------

    def update_provider_stats(
        self,
        provider_name: str,
        model: str,
        success: bool,
        response_time: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT success_count, fail_count, total_response_time "
                "FROM provider_stats WHERE provider_name = ? AND model = ?",
                (provider_name, model),
            ).fetchone()

            if row:
                s_count, f_count, t_time = row
                if success:
                    s_count += 1
                    t_time += response_time
                    status = "active"
                else:
                    f_count += 1
                    total = s_count + f_count
                    status = "degraded" if f_count / max(total, 1) > 0.5 else "active"

                conn.execute(
                    """UPDATE provider_stats
                       SET success_count=?, fail_count=?, total_response_time=?,
                           last_used=datetime('now'), last_error=?, status=?
                       WHERE provider_name=? AND model=?""",
                    (s_count, f_count, t_time, error, status, provider_name, model),
                )
            else:
                conn.execute(
                    """INSERT INTO provider_stats
                       (provider_name, model, success_count, fail_count,
                        total_response_time, last_error, status, last_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (
                        provider_name, model,
                        1 if success else 0,
                        0 if success else 1,
                        response_time if success else 0.0,
                        error,
                        "active" if success else "degraded",
                    ),
                )
            conn.commit()

    def get_all_provider_stats(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT provider_name, model, success_count, fail_count,
                      total_response_time, last_used, status
               FROM provider_stats ORDER BY success_count DESC"""
        ).fetchall()

        result = []
        for row in rows:
            prov, model, succ, fail, t_time, last_used, status = row
            total = succ + fail
            result.append({
                "provider": prov,
                "model": model,
                "success_count": succ,
                "fail_count": fail,
                "avg_response_time": t_time / succ if succ > 0 else 0.0,
                "success_rate": succ / total if total > 0 else 0.0,
                "last_used": last_used,
                "status": status,
            })
        return result

    def get_provider_stats(
        self, provider_name: str, model: str
    ) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT success_count, fail_count, total_response_time,
                      last_used, last_error, status
               FROM provider_stats WHERE provider_name=? AND model=?""",
            (provider_name, model),
        ).fetchone()

        if row:
            s, f, t, lu, le, st = row
            return {
                "success_count": s, "fail_count": f,
                "avg_response_time": t / s if s > 0 else 0.0,
                "last_used": lu, "last_error": le, "status": st,
            }
        return None

    # ------------------------------------------------------------------
    # Config state
    # ------------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO config_state (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))""",
                (key, json.dumps(value)),
            )
            conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM config_state WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    # ------------------------------------------------------------------
    # Update history
    # ------------------------------------------------------------------

    def add_update_record(
        self,
        from_version: str,
        to_version: str,
        status: str,
        error_log: Optional[str] = None,
    ) -> None:
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """INSERT INTO update_history
                   (from_version, to_version, status, error_log)
                   VALUES (?, ?, ?, ?)""",
                (from_version, to_version, status, error_log),
            )
            conn.commit()

    def get_database_size(self) -> int:
        return self.db_path.stat().st_size if self.db_path.exists() else 0

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_storage_lock = threading.Lock()
_storage: Optional[Storage] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = Storage()
    return _storage