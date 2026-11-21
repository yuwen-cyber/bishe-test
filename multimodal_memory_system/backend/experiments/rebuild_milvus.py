#!/usr/bin/env python3
"""从 MySQL + MinIO 重建 Milvus 向量索引（用于 schema 变更后回填数据）。

用法（在 backend 目录下）：
    python experiments/rebuild_milvus.py

说明：
- 不删除 / 不新增 MySQL 与 MinIO 数据，只重建 Milvus 中的向量。
- 读取 segments（含 is_demo）与 conversations 的 object key，从 MinIO 取回
  对应视频/音频字节，重新计算文本/音频/图像/情绪四路向量后写入 Milvus。
- 新增的音频/图像/情绪向量只存不查，与 upload.py 容错策略一致（单条失败不中断）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 让 `python experiments/rebuild_milvus.py` 时也能 import 到 backend 下的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    audio_encoder,
    embeddings,
    emotion_embedding,
    image_encoder,
    milvus_client,
    models,
    storage,
)
from app.config import settings
from app.database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rebuild_milvus")


def _minio_bytes(object_key: str | None) -> bytes:
    if not object_key:
        return b""
    client = storage.get_client()
    resp = client.get_object(settings.minio_bucket, object_key)
    try:
        return resp.read()
    finally:
        resp.close()


def main() -> int:
    db = SessionLocal()
    try:
        segs = (
            db.query(models.Segment, models.Conversation)
            .join(models.Conversation, models.Segment.conversation_id == models.Conversation.id)
            .order_by(models.Segment.is_demo.asc(), models.Segment.id.asc())
            .all()
        )
    finally:
        db.close()

    if not segs:
        logger.warning("MySQL 中没有可回填的 segment")
        return 0

    logger.info("共 %d 条 segment 待回填", len(segs))
    for module in (embeddings, audio_encoder, image_encoder, milvus_client):
        module.warmup()

    ok = 0
    for seg, conv in segs:
        try:
            video_bytes = _minio_bytes(conv.video_object_key)
            audio_bytes = _minio_bytes(conv.audio_object_key)

            text = seg.asr_text or ""
            emotion_label = seg.emotion or "中性"

            vec = embeddings.embed_text(text)
            audio_vec = audio_encoder.encode_audio(audio_bytes)
            image_vec = image_encoder.encode_video_keyframe(video_bytes)
            emo_vec = emotion_embedding.embed_emotion(emotion_label)

            milvus_client.insert_vector(
                seg.id,
                seg.conversation_id,
                text,
                emotion_label,
                int(seg.start_time.timestamp()),
                conv.video_object_key,
                conv.audio_object_key,
                vec,
                audio_vec,
                image_vec,
                emo_vec,
            )
            ok += 1
            logger.info(
                "[%d/%d] seg=%s demo=%s emotion=%s dims(text=%d,audio=%d,image=%d,emo=%d)",
                ok, len(segs), seg.id, seg.is_demo, emotion_label,
                len(vec), len(audio_vec), len(image_vec), len(emo_vec),
            )
        except Exception as exc:  # pragma: no cover - 单条失败不中断
            logger.warning("回填失败 seg=%s: %s", seg.id, exc)

    logger.info("完成：成功 %d / %d", ok, len(segs))
    return 0 if ok == len(segs) else 1


if __name__ == "__main__":
    sys.exit(main())