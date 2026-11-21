"""混合检索路由（创新点二：SQL 结构化预筛 + 向量语义召回）。

本文件为新增的独立模块，不修改现有 search.py 的基准/向量接口。

检索流程：
  1. SQL 路：MySQL 全文 MATCH...AGAINST 召回候选（带情绪/时间过滤）；
  2. 向量路：bge 文本编码 -> Milvus 语义召回（带标量过滤）；
  3. 融合：两路排名按加权 RRF（Reciprocal Rank Fusion）合并排序，
     向量路权重 2.0、SQL 路权重 1.0（与离线评测 hybrid_weighted 口径一致）；
  4. 回查 MySQL 取 is_demo 做真实录制优先的二级排序，返回前 top3。

RRF 只依赖两路排名、不比较分数尺度，避免全文相关性分与余弦相似度不可直接相加的问题。
"""
from __future__ import annotations

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

_MIN_SCORE = 0.5
_CAND_N = 50          # 每路召回候选数
_TOP_K = 3            # 最终返回条数
_RRF_K = 60
_W_SQL = 1.0
_W_VEC = 2.0

_BASE_SQL = (
    "SELECT s.id AS segment_id, s.conversation_id AS conversation_id, "
    "s.start_time AS start_time, s.emotion AS emotion, s.asr_text AS text, "
    "s.is_demo AS is_demo, c.video_object_key AS video_key, c.audio_object_key AS audio_key, "
    "MATCH(s.asr_text) AGAINST (:q IN NATURAL LANGUAGE MODE) AS score "
    "FROM segments s JOIN conversations c ON s.conversation_id = c.id "
)


def _sql_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _resolve_emotions(q: str, emotions: Optional[str]) -> list[str]:
    emo_list = [e.strip() for e in (emotions or "").split(",") if e.strip()]
    if not emo_list:
        inferred = emotion.infer_emotion_from_lexicon(q)
        if inferred:
            emo_list = [inferred]
    return emo_list


def _presigned(key: str | None) -> Optional[str]:
    if not key:
        return None
    try:
        return storage.presigned_url(key)
    except Exception:
        return None


@router.get("/hybrid/search", response_model=schemas.SearchResponse)
def hybrid_search(
    q: str = Query(..., min_length=1),
    emotions: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
):
    """SQL 全文召回 + 向量语义召回，加权 RRF 融合排序。"""
    t0 = time.perf_counter()
    emo_list = _resolve_emotions(q, emotions)
    low_info = query_expander.is_low_information(q)

    # ---- 路 1：SQL 全文召回 ----
    clauses: list[str] = []
    params: dict = {"q": q, "n": _CAND_N}
    if emo_list:
        clauses.append("s.emotion IN (" + ",".join(_sql_literal(e) for e in emo_list) + ")")
    if start:
        clauses.append("s.start_time >= :start")
        params["start"] = start
    if end:
        clauses.append("s.start_time <= :end")
        params["end"] = end

    if low_info:
        # ngram 最小分词 2 字，单字查询命中不了全文索引，退回 LIKE 包含匹配
        clauses.insert(0, "s.asr_text LIKE :kw")
        params["kw"] = f"%{q.strip()}%"
        where = "WHERE " + " AND ".join(clauses)
        sql_rows = db.execute(
            text(_BASE_SQL + where + " ORDER BY s.is_demo ASC, s.start_time DESC LIMIT :n"),
            params,
        ).mappings().all()
    else:
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql_rows = db.execute(
            text(_BASE_SQL + where + " HAVING score > 0 ORDER BY score DESC, s.is_demo ASC LIMIT :n"),
            params,
        ).mappings().all()

    sql_ids: list[int] = []
    meta: dict[int, dict] = {}
    for r in sql_rows:
        sid = int(r["segment_id"])
        sql_ids.append(sid)
        meta[sid] = {
            "conversation_id": int(r["conversation_id"]),
            "text": r["text"],
            "emotion": r["emotion"],
            "start_time": r["start_time"],
            "video_key": r["video_key"],
            "audio_key": r["audio_key"],
            "is_demo": bool(r["is_demo"]),
        }

    # ---- 路 2：向量语义召回 ----
    # 低信息查询的向量近邻基本是随机的，不参与融合，避免无关片段被 is_demo 顶到首位
    vec_ids: list[int] = []
    if not low_info:
        vec = embeddings.embed_text(q)
        hits = milvus_client.search_vectors(vec, top_k=_CAND_N, emotions=emo_list, start=start, end=end)
        for h in hits:
            sid = int(h["segment_id"])
            vec_ids.append(sid)
            if sid not in meta:
                meta[sid] = {
                    "conversation_id": int(h["conversation_id"]),
                    "text": h["text"],
                    "emotion": h["emotion"],
                    "start_time": datetime.fromtimestamp(h["start_time"]) if h.get("start_time") else None,
                    "video_key": h["video_key"],
                    "audio_key": h["audio_key"],
                    "is_demo": False,
                }

    # ---- 加权 RRF 融合 ----
    scores: dict[int, float] = {}
    for rank, sid in enumerate(sql_ids, start=1):
        scores[sid] = scores.get(sid, 0.0) + _W_SQL / (_RRF_K + rank)
    for rank, sid in enumerate(vec_ids, start=1):
        scores[sid] = scores.get(sid, 0.0) + _W_VEC / (_RRF_K + rank)

    fused = sorted(scores, key=lambda s: -scores[s])

    # 回查 MySQL 补全 is_demo（向量路候选可能未带 is_demo）
    if fused:
        id_list = ",".join(str(i) for i in fused)
        rows = db.execute(
            text(f"SELECT id, is_demo FROM segments WHERE id IN ({id_list})")
        ).mappings().all()
        demo_map = {int(r["id"]): bool(r["is_demo"]) for r in rows}
    else:
        demo_map = {}

    # 二级排序：真实录制(is_demo=False)优先，同组内按 RRF 分降序
    fused.sort(key=lambda sid: (demo_map.get(sid, meta.get(sid, {}).get("is_demo", False)), -scores[sid]))

    results: list[schemas.SearchResultItem] = []
    for sid in fused[:_TOP_K]:
        m = meta.get(sid, {})
        results.append(
            schemas.SearchResultItem(
                segment_id=sid,
                conversation_id=m.get("conversation_id", 0),
                timestamp=m.get("start_time"),
                emotion=m.get("emotion"),
                asr_text=m.get("text", ""),
                audio_url=_presigned(m.get("audio_key")),
                video_url=_presigned(m.get("video_key")),
                score=round(scores[sid], 4),
                is_demo=demo_map.get(sid, m.get("is_demo", False)),
                data_source=data_source.data_source(
                    sid, demo_map.get(sid, m.get("is_demo"))
                ),
            )
        )

    retrieval_time_ms = (time.perf_counter() - t0) * 1000
    return schemas.SearchResponse(retrieval_time_ms=retrieval_time_ms, total=len(results), results=results)
