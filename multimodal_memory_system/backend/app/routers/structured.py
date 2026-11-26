"""结构化精确查询路由（实验 3：基准方案的系统级能力展示）。

本文件为新增的独立模块，不修改现有 search.py / hybrid.py 的任何代码。

与 /api/baseline/search、/api/advanced/search 的区别：
  那两个接口都要求自然语言查询词 q，向量接口还带 0.5 相似度阈值，
  因此**无法表达"只按条件查"**（例如"13 点这一小时里情绪悲伤的片段"）。
  本接口补上纯条件查询能力：

      GET /api/structured/search?emotion=悲伤&start=...&end=...&keyword=累&limit=50

支持条件（可任意组合，组合即 AND）：
  emotion   情绪等值，多个用逗号分隔（喜悦,悲伤）
  start/end 时间范围（ISO 或 YYYY-MM-DD HH:MM:SS）
  keyword   片段文本包含该关键词（分词能力有限，属词面匹配兜底）
  min_duration_ms  片段所属会话时长下限（用于时长类条件查询）

排序：真实录制优先（is_demo=False），再按时间倒序，保证同一条件下结果稳定可复现。
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import schemas, storage
from ..database import get_db

router = APIRouter()

_MAX_LIMIT = 300

_BASE_SQL = (
    "SELECT s.id AS segment_id, s.conversation_id AS conversation_id, "
    "s.start_time AS start_time, s.emotion AS emotion, s.asr_text AS text, "
    "s.is_demo AS is_demo, c.video_object_key AS video_key, c.audio_object_key AS audio_key "
    "FROM segments s JOIN conversations c ON s.conversation_id = c.id "
)


def _sql_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _presigned(key: str | None) -> Optional[str]:
    if not key:
        return None
    try:
        return storage.presigned_url(key)
    except Exception:
        return None


@router.get("/structured/search", response_model=schemas.SearchResponse)
def structured_search(
    emotion: Optional[str] = Query(None, description="情绪等值，多个用逗号分隔"),
    emotions: Optional[str] = Query(None, description="同 emotion 的别名，兼容其余检索接口的复数写法"),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    keyword: Optional[str] = Query(None, description="片段文本包含该关键词"),
    min_duration_ms: Optional[int] = Query(None, ge=0, description="所属会话时长下限（毫秒）"),
    limit: int = Query(50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
):
    """纯结构化条件查询：情绪/时间/关键词/时长任意组合（AND），不依赖向量相似度。"""
    t0 = time.perf_counter()

    clauses: list[str] = []
    params: dict = {}

    # 同时接受 emotion（本接口原有）与 emotions（其余检索接口的写法），
    # 两者都传时取并集，避免客户端按错参数名后筛选被静默忽略。
    emo_raw = ",".join(x for x in (emotion, emotions) if x)
    emo_list = [e.strip() for e in emo_raw.split(",") if e.strip()]
    if emo_list:
        clauses.append("s.emotion IN (" + ",".join(_sql_literal(e) for e in emo_list) + ")")
    if start:
        clauses.append("s.start_time >= :start")
        params["start"] = start
    if end:
        clauses.append("s.start_time <= :end")
        params["end"] = end
    if keyword:
        clauses.append("s.asr_text LIKE :kw")
        params["kw"] = "%" + keyword + "%"
    if min_duration_ms is not None:
        clauses.append("c.duration_ms >= :dur")
        params["dur"] = int(min_duration_ms)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    # 先取总数（便于前端分页/展示"共命中 N 条"）
    total = int(
        db.execute(
            text(
                "SELECT COUNT(*) FROM segments s JOIN conversations c "
                "ON s.conversation_id = c.id" + where
            ),
            params,
        ).scalar()
        or 0
    )

    sql = text(
        _BASE_SQL + where +
        " ORDER BY s.is_demo ASC, s.start_time DESC LIMIT :limit"
    )
    rows = db.execute(sql, {**params, "limit": limit}).mappings().all()

    results = [
        schemas.SearchResultItem(
            segment_id=int(r["segment_id"]),
            conversation_id=int(r["conversation_id"]),
            timestamp=r["start_time"],
            emotion=r["emotion"],
            asr_text=r["text"],
            audio_url=_presigned(r["audio_key"]),
            video_url=_presigned(r["video_key"]),
            score=None,
            is_demo=bool(r["is_demo"]),
        )
        for r in rows
    ]

    retrieval_time_ms = (time.perf_counter() - t0) * 1000
    return schemas.SearchResponse(retrieval_time_ms=retrieval_time_ms, total=total, results=results)
