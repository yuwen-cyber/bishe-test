from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表，并确保 segments.asr_text 全文索引与 is_demo 列存在。"""
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() AND table_name = 'segments' "
                "AND index_name = 'ft_segments_asr_text'"
            )
        ).scalar()
        if not exists:
            conn.execute(
                text(
                    "CREATE FULLTEXT INDEX ft_segments_asr_text "
                    "ON segments (asr_text) WITH PARSER ngram"
                )
            )
        col_exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = 'segments' "
                "AND column_name = 'is_demo'"
            )
        ).scalar()
        if not col_exists:
            conn.execute(
                text("ALTER TABLE segments ADD COLUMN is_demo TINYINT(1) NOT NULL DEFAULT 0")
            )