from __future__ import annotations

import logging
from typing import Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .config import settings

logger = logging.getLogger(__name__)
_collection: Collection | None = None


def _connect() -> None:
    connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)


def _create_collection() -> Collection:
    dim = settings.embedding_dim
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="segment_id", dtype=DataType.INT64),
        FieldSchema(name="conversation_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="emotion", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="start_time", dtype=DataType.INT64),
        FieldSchema(name="video_key", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="audio_key", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="audio_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="emotion_embedding", dtype=DataType.FLOAT_VECTOR, dim=64),
    ]
    schema = CollectionSchema(fields, description="segment text embeddings + metadata")
    col = Collection(settings.milvus_collection, schema=schema)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    for field_name in ("embedding", "audio_embedding", "image_embedding", "emotion_embedding"):
        col.create_index(field_name=field_name, index_params=index_params)
    col.load()
    return col


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        _connect()
        if utility.has_collection(settings.milvus_collection):
            _collection = Collection(settings.milvus_collection)
        else:
            _collection = _create_collection()
    return _collection


def drop_collection() -> None:
    """删除集合（用于重建，schema 变更后旧集合需重建）。"""
    global _collection
    _connect()
    if utility.has_collection(settings.milvus_collection):
        utility.drop_collection(settings.milvus_collection)
    _collection = None


def insert_vector(
    segment_id: int,
    conversation_id: int,
    text: str,
    emotion: Optional[str],
    start_time: int,
    video_key: str,
    audio_key: str,
    embedding: list,
    audio_embedding: Optional[list] = None,
    image_embedding: Optional[list] = None,
    emotion_embedding: Optional[list] = None,
) -> None:
    # Milvus 向量字段不支持 null，缺省时用零向量占位
    audio_vec = list(audio_embedding) if audio_embedding is not None else [0.0] * 768
    image_vec = list(image_embedding) if image_embedding is not None else [0.0] * 512
    emo_vec = list(emotion_embedding) if emotion_embedding is not None else [0.0] * 64
    col = _get_collection()
    col.insert(
        [
            [segment_id],
            [conversation_id],
            [text],
            [emotion or ""],
            [int(start_time or 0)],
            [video_key or ""],
            [audio_key or ""],
            [embedding],
            [audio_vec],
            [image_vec],
            [emo_vec],
        ]
    )
    col.flush()


def _build_expr(
    emotions: Optional[list[str]] = None,
    start: Optional[object] = None,
    end: Optional[object] = None,
) -> Optional[str]:
    conds: list[str] = []
    if emotions:
        in_list = ", ".join(f'"{e}"' for e in emotions)
        conds.append(f"emotion in [{in_list}]")
    if start:
        conds.append(f"start_time >= {int(start.timestamp())}")
    if end:
        conds.append(f"start_time <= {int(end.timestamp())}")
    return " and ".join(conds) if conds else None


def search_vectors(
    embedding: list,
    top_k: int = 200,
    emotions: Optional[list[str]] = None,
    start: Optional[object] = None,
    end: Optional[object] = None,
) -> list[dict]:
    col = _get_collection()
    col.load()
    expr = _build_expr(emotions, start, end)
    # HNSW 要求 ef >= k(limit)：数据量增大、检索真正走索引后，固定 ef=64 会被服务端拒绝
    search_params = {"metric_type": "COSINE", "params": {"ef": max(64, int(top_k))}}
    results = col.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "segment_id",
            "conversation_id",
            "text",
            "emotion",
            "start_time",
            "video_key",
            "audio_key",
        ],
    )
    hits: list[dict] = []
    for batch in results:
        for hit in batch:
            hits.append(
                {
                    "segment_id": int(hit.entity.get("segment_id")),
                    "conversation_id": int(hit.entity.get("conversation_id")),
                    "text": hit.entity.get("text"),
                    "emotion": hit.entity.get("emotion"),
                    "start_time": int(hit.entity.get("start_time") or 0),
                    "video_key": hit.entity.get("video_key"),
                    "audio_key": hit.entity.get("audio_key"),
                    "distance": float(hit.distance),
                    "score": float(hit.distance),  # COSINE 下 distance 即余弦相似度，越大越相似
                }
            )
    return hits


def warmup() -> None:
    try:
        _get_collection()
        logger.info("Milvus collection ready: %s", settings.milvus_collection)
    except Exception as e:  # pragma: no cover
        logger.warning("Milvus not ready at startup: %s", e)


if __name__ == "__main__":
    # schema 变更后手动重建集合：python -m app.milvus_client
    drop_collection()
    logger.info("Collection dropped, will be recreated on next startup")