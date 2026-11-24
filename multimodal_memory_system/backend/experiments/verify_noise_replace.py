#!/usr/bin/env python3
"""校验噪声视频替换后的向量库一致性：

1. 可读行数是否为 2003；
2. 是否存在同一 segment_id 多行（上次 replace 曾出现此问题）；
3. 抽样检查噪声行的 image_embedding 是否有效且与正式行不同；
4. MinIO 对象大小是否与新切片一致。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\verify_noise_replace.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import milvus_client, models, storage  # noqa: E402
from app.database import SessionLocal  # noqa: E402

NOISE_DIR = EXPERIMENTS_DIR / "noise"
VIDEO_DIR = NOISE_DIR / "video"
MAPPING = EXPERIMENTS_DIR / "output" / "batch_import_mapping.json"


def main() -> int:
    col = milvus_client._get_collection()
    col.load()

    rows = col.query(expr="segment_id >= 0",
                     output_fields=["segment_id", "image_embedding"],
                     limit=16384)
    print(f"可读行数: {len(rows)}")
    ids = Counter(int(r["segment_id"]) for r in rows)
    dups = {k: v for k, v in ids.items() if v > 1}
    print(f"重复 segment_id: {len(dups)} 个" + (f"，示例 {list(dups.items())[:3]}" if dups else ""))

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    noise_ids = [int(m["db_segment_id"]) for m in mapping]

    by_id = {int(r["segment_id"]): r for r in rows}
    sample = [noise_ids[i] for i in (0, 300, 700, 1000, 1299) if noise_ids[i] in by_id]

    print("\n抽样噪声行：")
    for sid in sample:
        emb = by_id[sid].get("image_embedding")
        print(f"  seg {sid}: image_embedding 维度 {len(emb) if emb else 0}"
              f"，前3值 {[round(float(x), 4) for x in (emb[:3] if emb else [])]}")

    # 正式行对比
    formal_emb = by_id[1].get("image_embedding")
    if formal_emb and sample:
        a = by_id[sample[0]]["image_embedding"]
        same = all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, formal_emb))
        print(f"\n噪声首抽样与正式 seg1 向量完全相同: {same}（应为 False）")

    # MinIO 对象大小
    db = SessionLocal()
    client = storage.get_client()
    print("\nMinIO 抽样（噪声）：")
    for m in (mapping[0], mapping[650], mapping[1299]):
        sid = int(m["db_segment_id"])
        seg = db.query(models.Segment).filter(models.Segment.id == sid).first()
        if not seg:
            continue
        conv = (db.query(models.Conversation)
                .filter(models.Conversation.id == seg.conversation_id).first())
        if not conv or not conv.video_object_key:
            continue
        try:
            info = client.stat_object(storage.settings.minio_bucket, conv.video_object_key)
            local = VIDEO_DIR / f"noise_{int(m['noise_id']):04d}.mp4"
            print(f"  seg {sid}: MinIO {info.size} 字节｜本地 {local.stat().st_size} 字节"
                  f"｜{'一致' if info.size == local.stat().st_size else '不一致'}")
        except Exception as e:
            print(f"  seg {sid}: 读取失败 {type(e).__name__}: {e}")

    db.close()
    return 1 if dups else 0


if __name__ == "__main__":
    raise SystemExit(main())
