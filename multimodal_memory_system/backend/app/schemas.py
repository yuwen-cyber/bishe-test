from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    segment_id: int
    conversation_id: int
    timestamp: Optional[datetime] = None
    emotion: Optional[str] = None
    asr_text: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    score: Optional[float] = None
    is_demo: bool = False
    # 数据来源标记：formal(正式) / noise(噪声) / user(自录) / unknown，仅用于前端展示
    data_source: str = "formal"


class SearchResponse(BaseModel):
    retrieval_time_ms: float
    total: int
    results: List[SearchResultItem]


class UploadResponse(BaseModel):
    conversation_id: int
    segment_id: int
    emotion: Optional[str] = None
    asr_text: Optional[str] = None
    message: str