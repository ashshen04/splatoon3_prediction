"""SQLAlchemy engine and session factory.

Reads DATABASE_URL from the environment. Falls back to a local SQLite file
(splatoon3.db) when DATABASE_URL is not set, so the pipeline works without
a running Postgres instance during local development.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///splatoon3.db")

# psycopg2 requires postgresql:// but Railway sometimes emits postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # SQLite doesn't support connection pools the same way
    **({} if DATABASE_URL.startswith("sqlite") else {"pool_size": 5, "max_overflow": 10}),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session():
    """Context-manager-friendly session getter."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
