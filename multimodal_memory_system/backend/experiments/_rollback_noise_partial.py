#!/usr/bin/env python3
"""回滚本轮已导入的噪声（图像向量为 mock 的那批），清理 MySQL / Milvus / 映射文件。

只删 is_demo=True 且 segment_id 出现在 mapping 里的记录，不动正式 700 条与用户自录数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import milvus_client, models  # noqa: E402
from app.database import SessionLocal  # noqa: E402

OUTPUT_DIR = EXPERIMENTS_DIR / "output"
MAP_FILES = ["seed_noise_mapping.json", "batch_import_mapping.json"]


def main() -> int:
    ids: list[int] = []
    conv_ids: list[int] = []
    for name in MAP_FILES:
        p = OUTPUT_DIR / name
        if not p.exists():
            continue
        for m in json.loads(p.read_text(encoding="utf-8")):
            if m.get("db_segment_id"):
                ids.append(int(m["db_segment_id"]))
            if m.get("conversation_id"):
                conv_ids.append(int(m["conversation_id"]))
    if not ids:
        print("没有可回滚的记录")
        return 0

    print(f"待回滚 {len(ids)} 条：segment_id={ids}")

    db = SessionLocal()
    deleted = (
        db.query(models.Segment)
        .filter(models.Segment.id.in_(ids), models.Segment.is_demo.is_(True))
        .delete(synchronize_session=False)
    )
    conv_deleted = (
        db.query(models.Conversation)
        .filter(models.Conversation.id.in_(conv_ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    db.close()
    print(f"MySQL 删除 segments={deleted} conversations={conv_deleted}")

    col = milvus_client._get_collection()
    expr = "segment_id in [" + ", ".join(str(i) for i in ids) + "]"
    col.delete(expr=expr)
    col.flush()
    print(f"Milvus 已删除 {expr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
