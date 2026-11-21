#!/usr/bin/env python3
"""1300 条噪声批量直连导入（复用 app 的模型与存储模块，不改任何业务代码）。

与 POST /api/upload 的差异（仅为提速，向量口径完全一致）：
1. 跳过 ASR：文本本来就是 TTS 台词，plan 里已有，无需再跑 paraformer；
2. Milvus 批量 insert，每 N 条 flush 一次（接口版每条 flush，是主要耗时之一）；
3. 其余环节与接口完全一致：MinIO 双写 / 情绪融合预测 / bge 文本向量 /
   wav2vec2 音频向量 / CLIP 图像向量 / 情绪查找表向量 / MySQL 双表。

口径对齐说明：
- is_demo 一律 True（与正式 700 条一致，避免被 is_demo 优先排序顶到前排）；
- start_time 随机落在正式数据的时间窗（2026-11-11 ~ 2026-11-18），保证按时间过滤时
  噪声与正式数据混杂，而不是被轻易剔除；
- emotion 默认跑真模型（与正式 700 条同一口径）；--skip-emotion 可用标注值兜底。

用法（必须在 backend 目录下执行）：
    .venv\\Scripts\\python.exe experiments\\batch_import_noise.py --limit 3
    .venv\\Scripts\\python.exe experiments\\batch_import_noise.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import (  # noqa: E402
    audio_encoder,
    embeddings,
    emotion,
    emotion_embedding,
    image_encoder,
    milvus_client,
    models,
    storage,
)
from app.database import SessionLocal  # noqa: E402

NOISE_DIR = EXPERIMENTS_DIR / "noise"
PLAN_PATH = NOISE_DIR / "noise_plan.json"
AUDIO_DIR = NOISE_DIR / "audio"
VIDEO_DIR = NOISE_DIR / "video"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
MAPPING_PATH = OUTPUT_DIR / "batch_import_mapping.json"
FAILED_PATH = OUTPUT_DIR / "batch_import_failed.json"

# 与正式 700 条相同的时间窗
TIME_START = datetime(2026, 11, 11, 13, 0, 0)
TIME_END = datetime(2026, 11, 18, 21, 0, 0)


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json(path: Path, data) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--skip-emotion", action="store_true", help="直接用标注情绪，省去情绪模型推理")
    parser.add_argument("--batch-flush", type=int, default=40, help="每多少条 flush 一次 Milvus")
    args = parser.parse_args()

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if args.start or args.limit:
        end = None if not args.limit else args.start + args.limit
        plan = plan[args.start:end]

    # 两套 mapping 合并去重：走 HTTP 灌进去的那些也算已完成
    http_mapping = _load_json(OUTPUT_DIR / "seed_noise_mapping.json", [])
    batch_mapping = _load_json(MAPPING_PATH, [])
    done = {int(m["noise_id"]) for m in http_mapping} | {int(m["noise_id"]) for m in batch_mapping}
    todo = [p for p in plan if int(p["noise_id"]) not in done]

    print(f"计划 {len(plan)} 条｜已完成 {len(done)} 条｜本次待导入 {len(todo)} 条"
          f"｜情绪={'标注' if args.skip_emotion else '真模型'}")
    if not todo:
        return 0

    db = SessionLocal()
    col = milvus_client._get_collection()  # 复用业务代码的集合句柄，只做批量 insert
    random.seed(42)

    failed: list[dict] = []
    pending_rows: list[list] = []
    t_all = time.time()
    prof = {"minio": 0.0, "emotion": 0.0, "text_vec": 0.0, "audio_vec": 0.0, "image_vec": 0.0, "mysql": 0.0, "milvus": 0.0}

    def flush_milvus() -> None:
        nonlocal pending_rows, prof
        if not pending_rows:
            return
        t = time.time()
        col.insert(pending_rows)
        col.flush()
        prof["milvus"] += time.time() - t
        pending_rows = []

    for idx, item in enumerate(todo, start=1):
        nid = int(item["noise_id"])
        a_path = AUDIO_DIR / f"noise_{nid:04d}.mp3"
        v_path = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        if not a_path.exists() or not v_path.exists():
            failed.append({"noise_id": nid, "error": "媒体文件缺失"})
            continue

        audio_bytes = a_path.read_bytes()
        video_bytes = v_path.read_bytes()

        token = uuid.uuid4().hex
        video_key = f"videos/{token}.webm"
        audio_key = f"audios/{token}.webm"

        t = time.time()
        storage.upload_bytes(video_bytes, video_key, "video/mp4")
        storage.upload_bytes(audio_bytes, audio_key, "audio/mpeg")
        prof["minio"] += time.time() - t

        text = item["text"]  # 等价于真 ASR 结果

        t = time.time()
        if args.skip_emotion:
            emo_label, emo_conf = item["emotion"], None
        else:
            emo_label, emo_conf = emotion.predict(text, audio_bytes)
        prof["emotion"] += time.time() - t

        t = time.time()
        text_vec = embeddings.embed_text(text)
        prof["text_vec"] += time.time() - t

        t = time.time()
        audio_vec = audio_encoder.encode_audio(audio_bytes)
        prof["audio_vec"] += time.time() - t

        t = time.time()
        image_vec = image_encoder.encode_video_keyframe(video_bytes)
        prof["image_vec"] += time.time() - t

        emo_vec = emotion_embedding.embed_emotion(emo_label)
        duration_ms = int(round(float(item["duration_sec"]) * 1000))
        offset = random.randint(0, int((TIME_END - TIME_START).total_seconds()))
        start_dt = TIME_START + timedelta(seconds=offset)

        t = time.time()
        try:
            conv = models.Conversation(
                video_object_key=video_key,
                audio_object_key=audio_key,
                duration_ms=duration_ms,
            )
            db.add(conv)
            db.flush()
            seg = models.Segment(
                conversation_id=conv.id,
                start_time=start_dt,
                end_time=start_dt + timedelta(milliseconds=duration_ms),
                asr_text=text,
                emotion=emo_label,
                emotion_confidence=emo_conf,
                is_demo=True,
            )
            db.add(seg)
            db.commit()
            db.refresh(seg)
        except Exception as e:
            db.rollback()
            failed.append({"noise_id": nid, "error": f"mysql: {type(e).__name__}: {e}"})
            print(f"[{idx}/{len(todo)}] noise={nid} MySQL 失败，跳过", file=sys.stderr)
            continue
        prof["mysql"] += time.time() - t

        pending_rows.append({
            "segment_id": seg.id,
            "conversation_id": conv.id,
            "text": text,
            "emotion": emo_label or "",
            "start_time": int(start_dt.timestamp()),
            "video_key": video_key,
            "audio_key": audio_key,
            "embedding": text_vec,
            "audio_embedding": audio_vec,
            "image_embedding": image_vec,
            "emotion_embedding": emo_vec,
        })
        if len(pending_rows) >= args.batch_flush:
            flush_milvus()

        batch_mapping.append({
            "noise_id": nid,
            "group": item["group"],
            "labeled_emotion": item["emotion"],
            "db_segment_id": seg.id,
            "conversation_id": conv.id,
            "backend_emotion": emo_label,
            "start_time": start_dt.isoformat(),
            "duration_ms": duration_ms,
        })
        _save_json(MAPPING_PATH, batch_mapping)

        el = time.time() - t_all
        rate = idx / el
        eta_min = (len(todo) - idx) / rate / 60 if rate > 0 else 0
        print(f"[{idx:>4}/{len(todo)}] noise={nid} seg={seg.id} 标注={item['emotion']} "
              f"识别={emo_label} | {el/idx:.1f}s/条 ETA {eta_min:.0f}min", flush=True)
        if idx % 20 == 0:
            n = idx
            print("    分段均耗(秒): " + " ".join(
                f"{k}={v/n:.2f}" for k, v in prof.items() if k != "milvus"
            ), flush=True)

    flush_milvus()
    db.close()
    _save_json(FAILED_PATH, failed)

    el = time.time() - t_all
    print("\n========== 批量导入结束 ==========")
    print(f"成功 {len(todo) - len(failed)} 条，失败 {len(failed)} 条，总用时 {el/60:.1f} 分钟")
    print("分段耗时累计（秒）:", {k: round(v, 1) for k, v in prof.items()})
    print(f"映射 -> {MAPPING_PATH}")
    if failed:
        print(f"失败清单 -> {FAILED_PATH}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
