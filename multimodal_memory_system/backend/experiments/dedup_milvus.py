#!/usr/bin/env python3
"""清理 Milvus 中因 replace_video_and_rebuild.py 的 delete 未生效而残留的重复行。

背景：
    replace 脚本对每条 segment 执行 delete(expr) 后立即 insert，但 Milvus 的删除
    是墓碑标记、需要 flush 且时间戳晚于旧行才生效，结果同一 segment_id 存在 2 行，
    检索时同名片段占用多个返回名额，指标被严重拉低。

做法：
    1. 拉取全部行的主键 id 与 segment_id；
    2. 对重复的 segment_id，保留主键最大（最后插入）的一行，删除其余；
    3. 按主键 delete 后 flush，再回读校验每个 segment_id 只剩 1 行。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\dedup_milvus.py --dry-run
    .venv\\Scripts\\python.exe experiments\\dedup_milvus.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from pymilvus import Collection, connections  # noqa: E402

from app.config import settings  # noqa: E402

logging.disable(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)
    col = Collection(settings.milvus_collection)
    col.load()

    print(f"num_entities（含墓碑）: {col.num_entities}")

    rows = col.query(expr="segment_id >= 0", output_fields=["id", "segment_id"], limit=16384)
    print(f"实际可读行数: {len(rows)}")

    by_seg: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_seg[int(r["segment_id"])].append(int(r["id"]))

    dup_segs = {s: ids for s, ids in by_seg.items() if len(ids) > 1}
    print(f"重复的 segment_id 数量: {len(dup_segs)}")
    extra = sum(len(ids) - 1 for ids in dup_segs.values())
    print(f"需要删除的多余行数: {extra}")

    if args.dry_run:
        for s, ids in list(dup_segs.items())[:10]:
            print(f"  样例 seg {s}: 主键 {ids}")
        return 0

    # 每个 segment 只保留主键最大的一行
    to_delete = [pid for ids in dup_segs.values() for pid in sorted(ids)[:-1]]
    if not to_delete:
        print("没有重复行，无需清理")
        return 0

    for i in range(0, len(to_delete), 500):
        chunk = to_delete[i : i + 500]
        expr = "id in [" + ",".join(str(p) for p in chunk) + "]"
        col.delete(expr=expr)
    col.flush()
    print(f"已删除 {len(to_delete)} 行并 flush")

    rows2 = col.query(expr="segment_id >= 0", output_fields=["id", "segment_id"], limit=16384)
    by_seg2: dict[int, list[int]] = defaultdict(list)
    for r in rows2:
        by_seg2[int(r["segment_id"])].append(int(r["id"]))
    still_dup = {s: ids for s, ids in by_seg2.items() if len(ids) > 1}
    print(f"清理后可读行数: {len(rows2)}，仍重复的 segment_id: {len(still_dup)}")
    return 0 if not still_dup else 1


if __name__ == "__main__":
    raise SystemExit(main())
