"""Database engine, session factory and FTS5 schema management.

SQLite is used in its default file mode with one writer. Foreign keys are
enabled per connection. The FTS5 virtual table ``chunk_fts`` is an
*external-content* table backed by ``chunks`` (keyed on ``chunks.id``), and three
triggers keep it synchronized on insert/update/delete so the lexical index and
the relational rows can never drift apart within a transaction.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from rag_workbench.persistence.models import Base

# FTS5 DDL. The external-content table reads its columns from `chunks`; the
# triggers mirror every row change into the inverted index.
FTS_DDL: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        text,
        section_path,
        content='chunks',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunk_fts(rowid, text, section_path)
        VALUES (new.id, new.text, COALESCE(new.section_path, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text, section_path)
        VALUES ('delete', old.id, old.text, COALESCE(old.section_path, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text, section_path)
        VALUES ('delete', old.id, old.text, COALESCE(old.section_path, ''));
        INSERT INTO chunk_fts(rowid, text, section_path)
        VALUES (new.id, new.text, COALESCE(new.section_path, ''));
    END
    """,
)

# Rebuilds the whole index from the content table; useful after manual repair.
FTS_REBUILD = "INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _sqlite_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None
    raw = url[len(prefix) :]
    if raw in ("", ":memory:"):
        return None
    return Path(raw)


class Database:
    """Owns the SQLAlchemy engine and session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self.is_sqlite = _is_sqlite(url)

        connect_args: dict[str, object] = {}
        if self.is_sqlite:
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(url, echo=echo, connect_args=connect_args, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

        if self.is_sqlite:
            event.listen(self.engine, "connect", self._on_connect)

    @staticmethod
    def _on_connect(dbapi_connection, _record) -> None:  # noqa: ANN001 - sqlite3 callback
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_schema(self) -> None:
        """Create relational tables, the FTS5 table and its triggers."""

        path = _sqlite_path(self.url) if self.is_sqlite else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            for statement in FTS_DDL:
                connection.execute(text(statement))

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from the chunks table."""

        with self.engine.begin() as connection:
            connection.execute(text(FTS_REBUILD))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session, committing on success and rolling back on error."""

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
