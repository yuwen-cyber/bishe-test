from sqlalchemy import Boolean, Column, BigInteger, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=True)
    video_object_key = Column(String(512), nullable=False)
    audio_object_key = Column(String(512), nullable=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    segments = relationship("Segment", back_populates="conversation")


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = {"mysql_charset": "utf8mb4", "mysql_engine": "InnoDB"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversations.id"), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    asr_text = Column(Text, nullable=False)
    emotion = Column(String(32), nullable=True, index=True)
    emotion_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    is_demo = Column(Boolean, nullable=False, default=False, server_default="0")

    conversation = relationship("Conversation", back_populates="segments")