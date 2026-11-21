import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import (
    asr,
    audio_encoder,
    embeddings,
    emotion,
    emotion_embedding,
    image_encoder,
    milvus_client,
    models,
    schemas,
    storage,
)
from ..database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=schemas.UploadResponse)
async def upload(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    duration_ms: int = Form(None),
    is_demo: bool = Form(False),
    db: Session = Depends(get_db),
):
    video_bytes = await video.read()
    audio_bytes = await audio.read()
    if not video_bytes or not audio_bytes:
        raise HTTPException(status_code=400, detail="video 与 audio 文件均不能为空")

    token = uuid.uuid4().hex
    video_key = f"videos/{token}.webm"
    audio_key = f"audios/{token}.webm"

    storage.upload_bytes(video_bytes, video_key, video.content_type or "video/webm")
    storage.upload_bytes(audio_bytes, audio_key, audio.content_type or "audio/webm")

    # 1) ASR
    text = asr.transcribe(audio_bytes)
    # 2) 情绪分析（文本 + 音频语音两路融合）
    emotion_label, emotion_conf = emotion.predict(text, audio_bytes)
    # 3) 文本向量化
    vec = embeddings.embed_text(text)

    now = datetime.now()
    # 4) 写 MySQL
    conv = models.Conversation(
        video_object_key=video_key,
        audio_object_key=audio_key,
        duration_ms=duration_ms,
    )
    db.add(conv)
    db.flush()

    seg = models.Segment(
        conversation_id=conv.id,
        start_time=now,
        end_time=now + timedelta(milliseconds=duration_ms or 0),
        asr_text=text,
        emotion=emotion_label,
        emotion_confidence=emotion_conf,
        is_demo=is_demo,
    )
    db.add(seg)
    db.commit()
    db.refresh(seg)

    # 5) 写 Milvus（尽力而为，失败不影响上传结果）
    try:
        audio_vec = audio_encoder.encode_audio(audio_bytes)
        image_vec = image_encoder.encode_video_keyframe(video_bytes)
        emo_vec = emotion_embedding.embed_emotion(emotion_label)
        milvus_client.insert_vector(
            seg.id, conv.id, text, emotion_label,
            int(now.timestamp()), video_key, audio_key, vec,
            audio_vec, image_vec, emo_vec,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Milvus insert failed for segment %s: %s", seg.id, e)

    return schemas.UploadResponse(
        conversation_id=conv.id,
        segment_id=seg.id,
        emotion=emotion_label,
        asr_text=text,
        message="对话已保存",
    )