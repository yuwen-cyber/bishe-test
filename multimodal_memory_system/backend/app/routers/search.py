import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import (data_source, embeddings, emotion, milvus_client,
                query_expander, schemas, storage)
from ..database import get_db

router = APIRouter()

# 余弦相似度下限：低于该值的命中视为不相关，不再硬凑 top3；
# 凑不满 3 条就返回更少，避免拿跨情绪的低相关片段充当第三条。
_MIN_SCORE = 0.5

_BASE_SQL = (
    "SELECT s.id, s.conversation_id, s.start_time, s.emotion, s.asr_text, s.is_demo, "
    "c.video_object_key, c.audio_object_key "
    "FROM segments s JOIN conversations c ON s.conversation_id = c.id "
)


def _parse_emotions(emotions: Optional[str]) -> list[str]:
    if not emotions:
        return []
    return [e.strip() for e in emotions.split(",") if e.strip()]


def _resolve_emotions(q: str, emotions: Optional[str]) -> list[str]:
    """合并显式情绪筛选与查询词推断的情绪标签（情绪一致检索）。

    用户未手动选情绪时，若查询词命中情感词典（如「开心/高兴」→喜悦），
    则自动按其对应情绪过滤，保证「开心」「高兴」这类同义情绪检索结果一致。
    """
    emo_list = _parse_emotions(emotions)
    if not emo_list:
        inferred = emotion.infer_emotion_from_lexicon(q)
        if inferred:
            emo_list = [inferred]
    return emo_list


def _sql_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _presigned(key: str) -> Optional[str]:
    try:
        return storage.presigned_url(key)
    except Exception:  # pragma: no cover
        return None


def _item_from_row(row, score: Optional[float] = None) -> schemas.SearchResultItem:
    return schemas.SearchResultItem(
        segment_id=row["id"],
        conversation_id=row["conversation_id"],
        timestamp=row["start_time"],
        emotion=row["emotion"],
        asr_text=row["asr_text"],
        audio_url=_presigned(row["audio_object_key"]),
        video_url=_presigned(row["video_object_key"]),
        score=score,
        is_demo=bool(row["is_demo"]),
        data_source=data_source.data_source(row["id"], bool(row["is_demo"])),
    )


def _build_filter(emotions: list[str], start: Optional[datetime], end: Optional[datetime]):
    clauses: list[str] = []
    params: dict = {}
    if emotions:
        # 情绪标签来源于固定集合并经过 SQL 转义，安全内联
        clauses.append("s.emotion IN (" + ",".join(_sql_literal(e) for e in emotions) + ")")
    if start:
        clauses.append("s.start_time >= :start")
        params["start"] = start
    if end:
        clauses.append("s.start_time <= :end")
        params["end"] = end
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


@router.get("/baseline/search", response_model=schemas.SearchResponse)
def baseline_search(
    q: str = Query(..., min_length=1),
    emotions: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
):
    """MySQL 全文检索（MATCH...AGAINST，ngram 解析器）+ 情绪/时间过滤。

    单字等低信息查询不会被 ngram（最小 2 字）切出 token，退回 LIKE 包含匹配。
    """
    t0 = time.perf_counter()
    emo_list = _resolve_emotions(q, emotions)
    where, params = _build_filter(emo_list, start, end)

    if query_expander.is_low_information(q):
        where = "WHERE s.asr_text LIKE :kw" + where
        params["kw"] = f"%{q.strip()}%"
        sql = text(_BASE_SQL + where + " ORDER BY s.is_demo ASC, s.start_time DESC LIMIT 3")
    else:
        match_clause = f"MATCH(s.asr_text) AGAINST ({_sql_literal(q)} IN NATURAL LANGUAGE MODE)"
        where = f"WHERE {match_clause}{where}"
        sql = text(_BASE_SQL + where + f" ORDER BY s.is_demo ASC, {match_clause} DESC LIMIT 3")

    rows = db.execute(sql, params).mappings().all()
    retrieval_time_ms = (time.perf_counter() - t0) * 1000

    results = [_item_from_row(r) for r in rows]
    return schemas.SearchResponse(retrieval_time_ms=retrieval_time_ms, total=len(results), results=results)


@router.get("/advanced/search", response_model=schemas.SearchResponse)
def advanced_search(
    q: str = Query(..., min_length=1),
    emotions: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    expand: str = Query("none", pattern="^(none|lexicon|llm)$"),
    db: Session = Depends(get_db),
):
    """Milvus 向量检索 + 标量过滤（情绪/时间），用 is_demo 做真实录制优先排序。

    expand 参数用于消融实验：在向量化之前先扩展查询词（none/lexicon/llm）。
    """
    t0 = time.perf_counter()
    emo_list = _resolve_emotions(q, emotions)

    # 低信息查询（如单个「了」）的向量近邻是随机的，返回空比返回无关片段更诚实
    if query_expander.is_low_information(q):
        retrieval_time_ms = (time.perf_counter() - t0) * 1000
        return schemas.SearchResponse(retrieval_time_ms=retrieval_time_ms, total=0, results=[])

    q = query_expander.expand_query(q, mode=expand)
    vec = embeddings.embed_text(q)
    hits = milvus_client.search_vectors(vec, top_k=200, emotions=emo_list, start=start, end=end)

    # 来源优先级：真实录制(is_demo=False)排在评测注入数据(is_demo=True)之前；
    # 同组内仍按 cosine 相似度降序，保持纯向量检索的排序语义。
    ids = [int(h["segment_id"]) for h in hits]
    demo_map: dict[int, bool] = {}
    if ids:
        id_list = ",".join(str(i) for i in ids)
        rows = db.execute(
            text(f"SELECT id, is_demo FROM segments WHERE id IN ({id_list})")
        ).mappings().all()
        demo_map = {int(r["id"]): bool(r["is_demo"]) for r in rows}

    # 先按相似度下限过滤，再让真实录制优先：
    # 否则不相关片段只要 is_demo=0 就会被无条件顶到首位。
    hits = [h for h in hits if h["score"] >= _MIN_SCORE]
    hits.sort(key=lambda h: (demo_map.get(int(h["segment_id"]), False), -h["score"]))
    retrieval_time_ms = (time.perf_counter() - t0) * 1000

    results = []
    for h in hits[:3]:
        results.append(
            schemas.SearchResultItem(
                segment_id=h["segment_id"],
                conversation_id=h["conversation_id"],
                timestamp=datetime.fromtimestamp(h["start_time"]) if h.get("start_time") else None,
                emotion=h.get("emotion"),
                asr_text=h["text"],
                audio_url=_presigned(h["audio_key"]),
                video_url=_presigned(h["video_key"]),
                score=h["score"],
                is_demo=demo_map.get(int(h["segment_id"]), False),
                data_source=data_source.data_source(
                    h["segment_id"], demo_map.get(int(h["segment_id"]))
                ),
            )
        )
    return schemas.SearchResponse(retrieval_time_ms=retrieval_time_ms, total=len(results), results=results)