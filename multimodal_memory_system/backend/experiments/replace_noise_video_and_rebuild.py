#!/usr/bin/env python3
"""把 1300 条噪声视频换成网上下载的无人物素材，并重算 CLIP 图像向量。

与 replace_video_and_rebuild.py 同思路，但目标是噪声数据：
    1. noise_XXXX.mp4（已重切）覆盖 MinIO 同 key 对象；
    2. 用真实 CLIP 重算 image_embedding（脚本内直接调 _encode_real，绕过
       .env 的 USE_REAL_IMAGE=false，避免影响线上后端）；
    3. Milvus 按 segment_id delete 后 insert 同一行，其余向量沿用原值。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\replace_noise_video_and_rebuild.py --limit 5
    .venv\\Scripts\\python.exe experiments\\replace_noise_video_and_rebuild.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import image_encoder, milvus_client, models, storage  # noqa: E402
from app.database import SessionLocal  # noqa: E402

NOISE_DIR = EXPERIMENTS_DIR / "noise"
VIDEO_DIR = NOISE_DIR / "video"
MAPPING = EXPERIMENTS_DIR / "output" / "batch_import_mapping.json"
PROGRESS = EXPERIMENTS_DIR / "output" / "replace_noise_progress.json"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"

QUERY_FIELDS = [
    "segment_id", "conversation_id", "text", "emotion", "start_time",
    "video_key", "audio_key", "embedding", "audio_embedding", "emotion_embedding",
]

FLUSH_EVERY = 40


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    print(f"噪声映射 {len(mapping)} 条，首条 db_segment_id={mapping[0]['db_segment_id']}")

    done: list[int] = []
    if PROGRESS.exists():
        try:
            done = json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", [])
        except Exception:
            done = []
    done_set = set(done)

    targets = mapping[args.start:]
    if args.limit:
        targets = targets[: args.limit]

    db = SessionLocal()
    col = milvus_client._get_collection()

    ok = skip = fail = 0
    t0 = time.time()

    for i, m in enumerate(targets, start=1):
        nid = int(m["noise_id"])
        sid = int(m["db_segment_id"])
        if sid in done_set:
            skip += 1
            continue

        video_path = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        if not video_path.exists():
            print(f"[{i}] nid {nid}: 缺少 {video_path.name}", file=sys.stderr)
            fail += 1
            continue

        seg = db.query(models.Segment).filter(models.Segment.id == sid).first()
        if seg is None:
            print(f"[{i}] db seg {sid} 不存在", file=sys.stderr)
            fail += 1
            continue
        conv = (db.query(models.Conversation)
                .filter(models.Conversation.id == seg.conversation_id).first())
        if conv is None or not conv.video_object_key:
            print(f"[{i}] db seg {sid} 无 conversation/video_key", file=sys.stderr)
            fail += 1
            continue

        video_bytes = video_path.read_bytes()

        # 1) MinIO 同 key 覆盖
        storage.upload_bytes(video_bytes, conv.video_object_key, "video/mp4")

        # 2) 取原行
        rows = col.query(expr=f"segment_id == {sid}", output_fields=QUERY_FIELDS)
        if not rows:
            print(f"[{i}] db seg {sid} 在 Milvus 中不存在", file=sys.stderr)
            fail += 1
            continue
        row = rows[0]

        # 3) 真实 CLIP 重算图像向量
        try:
            new_img = image_encoder._encode_real(video_bytes)
        except Exception as e:
            print(f"[{i}] nid {nid}: CLIP 编码失败 {type(e).__name__}: {e}", file=sys.stderr)
            fail += 1
            continue

        # 4) delete + insert
        col.delete(expr=f"segment_id == {sid}")
        col.insert([{
            "segment_id": sid,
            "conversation_id": int(row["conversation_id"]),
            "text": row["text"],
            "emotion": row.get("emotion") or "",
            "start_time": int(row["start_time"]),
            "video_key": row["video_key"],
            "audio_key": row["audio_key"],
            "embedding": row["embedding"],
            "audio_embedding": row["audio_embedding"],
            "image_embedding": new_img,
            "emotion_embedding": row["emotion_embedding"],
        }])

        ok += 1
        done.append(sid)
        done_set.add(sid)

        if ok % FLUSH_EVERY == 0:
            col.flush()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            PROGRESS.write_text(json.dumps({"done": done}, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"[{i}/{len(targets)}] seg {sid} | {el/ok:.2f}s/条 | 预计剩余 "
                  f"{(len(targets)-i)*el/ok/60:.1f} 分钟", flush=True)

    col.flush()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"done": done}, ensure_ascii=False), encoding="utf-8")
    db.close()
    print(f"\n完成：替换 {ok} 条，跳过 {skip} 条，失败 {fail} 条｜用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
