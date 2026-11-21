#!/usr/bin/env python3
"""把正式 700 条的占位视频替换为真实情绪切片，并重算 CLIP 图像向量。

为什么必须重算向量：
    Milvus 里存的是 CLIP 对"中间帧"的编码结果，换视频文件不等于换向量。
    原 700 条的图像向量不仅来自同一段 placeholder（700 条完全相同），
    还因 transformers 5.x 接口变更全部回退成了 md5 伪向量。

做法（保持 segment_id 不变，因此不动 query_set_db.json 的真值映射）：
    1. 用 MinIO 同一个 video_object_key 覆盖对象（前端回放立即变成真实切片）；
    2. 对该视频重算 image_embedding；
    3. Milvus 按 segment_id delete 后 insert 同一行（其余向量沿用原值）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\replace_video_and_rebuild.py --limit 3
    .venv\\Scripts\\python.exe experiments\\replace_video_and_rebuild.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import image_encoder, milvus_client, models, storage  # noqa: E402
from app.database import SessionLocal  # noqa: E402

REAL_VIDEO_DIR = EXPERIMENTS_DIR / "real_videos"
DATASET = EXPERIMENTS_DIR / "segments_dataset.json"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
PROGRESS = OUTPUT_DIR / "replace_video_progress.json"

QUERY_FIELDS = [
    "segment_id", "conversation_id", "text", "emotion", "start_time",
    "video_key", "audio_key", "embedding", "audio_embedding", "emotion_embedding",
]


def _norm(text: str) -> str:
    """去掉标点与空白后比较：ASR 输出的标点常与数据集原文不一致。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", (text or ""))


def _similar(a: str, b: str) -> float:
    """ASR 常有错字（如 事→是），按字符序列相似度容忍少量差异。"""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=1)
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    # 入库按 dataset 列表顺序进行：db id i 对应 dataset[i-1]。
    # dataset 自身缺 segment_id 438，所以不能用 segment_id 反查，必须按索引取；
    # 切片文件名同样按 segment_id 命名（generate_real_slices.py 的输出）。
    ds_by_index = {i: x for i, x in enumerate(dataset)}

    done: list[int] = []
    if PROGRESS.exists():
        try:
            done = json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", [])
        except Exception:
            done = []

    db = SessionLocal()
    col = milvus_client._get_collection()

    # 正式 700 条 = db id 1..700 且 is_demo=True
    targets = (
        db.query(models.Segment)
        .filter(models.Segment.id >= args.start_id, models.Segment.id <= 700,
                models.Segment.is_demo.is_(True))
        .order_by(models.Segment.id)
        .all()
    )
    if args.limit:
        targets = targets[: args.limit]

    ok = skip = fail = 0
    t0 = time.time()
    for i, seg in enumerate(targets, start=1):
        sid = int(seg.id)
        if sid in done:
            skip += 1
            continue

        idx = sid - 1  # dataset 列表索引
        item = ds_by_index.get(idx)
        if item is None:
            print(f"[{i}] seg {sid} 在 dataset 中无对应记录，跳过", file=sys.stderr)
            fail += 1
            continue
        if _similar(item["asr_text"], seg.asr_text) < 0.75:
            print(f"[{i}] seg {sid} 文本与 dataset 相似度过低，跳过（需人工核对）", file=sys.stderr)
            fail += 1
            continue

        ds_seg_id = int(item["segment_id"])
        video_path = REAL_VIDEO_DIR / f"seg_{ds_seg_id:04d}.mp4"
        if not video_path.exists():
            print(f"[{i}] seg {sid} 缺少切片 {video_path.name}", file=sys.stderr)
            fail += 1
            continue

        conv = db.query(models.Conversation).filter(models.Conversation.id == seg.conversation_id).first()
        if conv is None or not conv.video_object_key:
            fail += 1
            print(f"[{i}] seg {sid} 找不到 conversation/video_key", file=sys.stderr)
            continue

        video_bytes = video_path.read_bytes()

        # 1) MinIO 同 key 覆盖
        storage.upload_bytes(video_bytes, conv.video_object_key, "video/mp4")

        # 2) 取原行（保留音频/文本/情绪向量）
        rows = col.query(expr=f"segment_id == {sid}", output_fields=QUERY_FIELDS)
        if not rows:
            # 库里缺这一行，用原视频?? 无法恢复其他向量，跳过并记录
            fail += 1
            print(f"[{i}] seg {sid} 在 Milvus 中不存在，跳过", file=sys.stderr)
            continue
        row = rows[0]

        # 3) 重算图像向量
        new_img = image_encoder.encode_video_keyframe(video_bytes)

        # 4) 替换该行
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
        col.flush()

        done.append(sid)
        ok += 1
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROGRESS.write_text(json.dumps({"done": done}, ensure_ascii=False), encoding="utf-8")

        if ok % 20 == 0:
            el = time.time() - t0
            print(f"[{i}/{len(targets)}] seg {sid} 已替换 | {el/ok:.1f}s/条", flush=True)

    db.close()
    print(f"\n完成：替换 {ok} 条，跳过 {skip} 条，失败 {fail} 条")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
