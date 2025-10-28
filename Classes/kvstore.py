import sqlite3
from typing import Optional

class KV :
    
    def __init__(self, path: str = './storage/kv/db.sqlite3') -> None:
        self.conn = sqlite3.connect(path)
        
    def create_table (self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self.conn.commit()
    
    def set (self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO kv VALUES (?, ?)
            """,
            (key, value)
        )
        self.conn.commit()
    
    def get (self, key: str) -> Optional[list[str]]:
        cur = self.conn.execute(
            """
            SELECT value FROM kv WHERE key=?
            """,
            (key,)
        )
        self.conn.commit()
        row = cur.fetchone()
        if row:
            return row[0] or None
    
    def keys (self) -> Optional[list[str]]:
        cur = self.conn.execute(
            """
            SELECT * FROM kv
            """
        )
        self.conn.commit()
        row = cur.fetchall()
        
        if row:
            return row or None
    
    def close (self) -> None:
        self.conn.close()