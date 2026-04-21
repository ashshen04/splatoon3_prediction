"""SQLAlchemy engine and session factory.

Reads DATABASE_URL from the environment. Falls back to a local SQLite file
(splatoon3.db) when DATABASE_URL is not set, so the pipeline works without
a running Postgres instance during local development.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///splatoon3.db")

# psycopg2 requires postgresql:// but Railway sometimes emits postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    **({} if DATABASE_URL.startswith("sqlite") else {"pool_size": 5, "max_overflow": 10}),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def check_db_connection() -> None:
    """Verify the DB is reachable. Raises with a clear message if not."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        logger.error(
            "Cannot connect to database at %s — is Postgres running?",
            _redacted_url(),
        )
        raise RuntimeError(f"Database connection failed: {e}") from e
    except SQLAlchemyError as e:
        logger.error("Database error while connecting: %s", e)
        raise

    logger.info("Connected to database: %s", _redacted_url())


def _redacted_url() -> str:
    """Return the DATABASE_URL with password hidden for safe logging."""
    url = engine.url
    if url.password:
        return str(url).replace(url.password, "***")
    return str(url)
