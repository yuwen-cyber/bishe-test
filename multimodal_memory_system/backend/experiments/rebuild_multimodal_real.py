#!/usr/bin/env python3
"""用真实多模态编码器重建 Milvus 向量（路线 B：完整模态消融前置）。

与 rebuild_milvus.py 的区别：
- 文本：bge-small-zh（真模型，复用 app.embeddings）
- 音频：Wav2Vec2-base（真模型，复用 app.audio_encoder._encode_real）
- 图像：CLIP ViT-B/32（真模型；本脚本内用 transformers 5.x 兼容取法
        取 out.pooler_output，不调用 app.image_encoder._encode_real，避免改业务代码）
- 情绪：固定查找表（复用 app.emotion_embedding）

流程：首批先 drop 旧 Milvus 集合（清掉之前的 mock 占位向量），再从 MySQL + MinIO
逐条取回音视频字节，真编码后重新写入。MySQL / MinIO 数据不删不改。

内存说明：本机 16GB，三模型常驻 + CPU 推理较吃内存，因此：
- 限制 torch 线程数，降低每线程工作内存副本；
- 支持 --start/--limit 分段，配合外层分批重启进程释放累积碎片；
- 每条编码后主动回收张量。

用法（backend 目录）：
    # 首批：清空集合，从头处理（可先 --limit 5 试跑）
    .venv\\Scripts\\python.exe experiments\\rebuild_multimodal_real.py --limit 5
    # 全量首批（清空集合）
    .venv\\Scripts\\python.exe experiments\\rebuild_multimodal_real.py
    # 分段续跑（不清空已有集合）：从索引 40 起处理 60 条
    .venv\\Scripts\\python.exe experiments\\rebuild_multimodal_real.py --start 40 --limit 60 --keep-collection
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# 必须在 import torch 前限制线程，降低 CPU 推理内存副本
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TORCH_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (  # noqa: E402
    audio_encoder,
    embeddings,
    emotion_embedding,
    image_encoder,
    milvus_client,
    models,
    storage,
)
from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rebuild_real")

IMAGE_SIZE = 224
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FAILED_PATH = OUTPUT_DIR / "rebuild_real_failed.json"

_clip_model = None


def _load_clip():
    global _clip_model
    if _clip_model is None:
        from transformers import CLIPModel

        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model


def encode_image_compat(video_bytes: bytes) -> list[float]:
    """CLIP 关键帧编码（transformers 5.x 兼容版）。

    复用 app.image_encoder 的抽帧与归一化常量；模型输出取 pooler_output。
    """
    model = _load_clip()
    raw_path = image_encoder._extract_keyframe_rgb(video_bytes)
    try:
        raw = np.fromfile(raw_path, dtype=np.uint8)
        img = raw.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)  # (3,224,224)
        tensor = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=tensor)
        feat = out.pooler_output.squeeze(0)
        feat = feat / max(float(torch.norm(feat)), 1e-12)
        result = feat.tolist()
        del tensor, out, feat, img, raw
        return result
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


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
    parser = argparse.ArgumentParser(description="真实编码器重建 Milvus 向量")
    parser.add_argument("--start", type=int, default=0, help="从第几条（0 基索引）开始")
    parser.add_argument("--limit", type=int, default=0, help="本次处理条数（0=到末尾）")
    parser.add_argument("--keep-collection", action="store_true", help="不 drop 旧集合（分段续跑用）")
    parser.add_argument("--ids-file", default="", help="JSON 数组文件，仅处理列出的 segment_id（补跑缺失用，避免重复入库）")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        segs = (
            db.query(models.Segment, models.Conversation)
            .join(models.Conversation, models.Segment.conversation_id == models.Conversation.id)
            .order_by(models.Segment.id.asc())
            .all()
        )
    finally:
        db.close()

    if not segs:
        logger.warning("MySQL 中没有可回填的 segment")
        return 0

    # 按灌库映射白名单过滤，排除不在 700 条真值内的早期测试残留（不删 MySQL 数据）
    mapping_path = OUTPUT_DIR / "seed_mapping_real.json"
    if mapping_path.exists():
        known = {x["db_segment_id"] for x in json.loads(mapping_path.read_text(encoding="utf-8"))}
        before = len(segs)
        segs = [(s, c) for s, c in segs if s.id in known]
        logger.info("白名单过滤：%d -> %d 条（排除 %d 条非真值残留）", before, len(segs), before - len(segs))

    if args.ids_file:
        want = {int(x) for x in json.loads(Path(args.ids_file).read_text(encoding="utf-8"))}
        before_ids = len(segs)
        segs = [(s, c) for s, c in segs if s.id in want]
        logger.info("按 ids-file 过滤：%d -> %d 条（目标 %d）", before_ids, len(segs), len(want))

    segs = segs[args.start:]
    if args.limit > 0:
        segs = segs[: args.limit]

    logger.info("本次处理 %d 条（起始索引 %d）", len(segs), args.start)

    # 触发文本 / 音频模型加载
    embeddings.warmup()
    audio_encoder.warmup()
    _load_clip()
    logger.info("模型加载完成（bge / Wav2Vec2 / CLIP），torch 线程数=%d", torch.get_num_threads())

    if not args.keep_collection:
        milvus_client.drop_collection()
        logger.info("旧 Milvus 集合已清空，即将重建")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[dict] = []
    ok = 0
    t_start = time.perf_counter()

    for idx, (seg, conv) in enumerate(segs, start=1):
        try:
            video_bytes = _minio_bytes(conv.video_object_key)
            audio_bytes = _minio_bytes(conv.audio_object_key)

            text = seg.asr_text or ""
            emotion_label = seg.emotion or "中性"

            vec = embeddings.embed_text(text)
            audio_vec = audio_encoder._encode_real(audio_bytes)
            image_vec = encode_image_compat(video_bytes)
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
            del video_bytes, audio_bytes, vec, audio_vec, image_vec, emo_vec

            if idx % 10 == 0:
                gc.collect()
            if idx % 20 == 0 or idx == len(segs):
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "[%d/%d] seg=%s emotion=%s 用时%.0fs",
                    idx, len(segs), seg.id, emotion_label, elapsed,
                )
        except Exception as exc:  # 单条失败不中断
            logger.warning("重建失败 seg=%s: %s", seg.id, exc)
            failed.append({"segment_id": seg.id, "error": str(exc)})
            gc.collect()

    FAILED_PATH.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完成：成功 %d / %d，失败 %d", ok, len(segs), len(failed))
    logger.info("失败清单 -> %s", FAILED_PATH)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
