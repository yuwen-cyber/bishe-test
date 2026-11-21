#!/usr/bin/env python3
"""清理人工测试上传的残留记录（is_demo=0），避免污染检索结果与答辩演示。

背景：700 条正式数据集入库时 is_demo=1；你在「对话录制」页手动上传测试句时
is_demo=0。search.py / hybrid.py 的排序规则是「is_demo=0 优先」，导致几条测试
句永远占据检索结果前排，把正式数据压下去。

安全约束（硬编码，不可通过参数绕过）：
1. 只处理 is_demo=0 的记录；目标里只要出现一条 is_demo=1，立即中止，不做任何删除。
2. 只删指定 id 列表内的记录，不做范围删除、不做模糊匹配。
3. 删除前把完整记录（含音视频 object key）导出到
   experiments/output/test_records_backup_<时间>.json，可凭 key 重新入库。
4. MinIO 里的音视频原件不删，保证可回滚。
5. 同时清理 Milvus 里对应的向量，并报告 Milvus 中不属于 MySQL 的孤儿向量。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\clean_test_records.py --dry-run   # 只预览，不删
    .venv\\Scripts\\python.exe experiments\\clean_test_records.py             # 真正执行
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymilvus import Collection, connections, utility  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import models  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_IDS = "701,702,703"


def _load_mysql(session, ids: list[int]) -> list[dict]:
    rows = (
        session.query(models.Segment, models.Conversation)
        .join(models.Conversation, models.Segment.conversation_id == models.Conversation.id)
        .filter(models.Segment.id.in_(ids))
        .order_by(models.Segment.id.asc())
        .all()
    )
    out: list[dict] = []
    for seg, conv in rows:
        out.append(
            {
                "segment_id": int(seg.id),
                "conversation_id": int(conv.id),
                "is_demo": bool(seg.is_demo),
                "start_time": seg.start_time.isoformat() if seg.start_time else None,
                "asr_text": seg.asr_text,
                "emotion": seg.emotion,
                "emotion_confidence": float(seg.emotion_confidence or 0),
                "duration_ms": int(conv.duration_ms or 0),
                "video_object_key": conv.video_object_key,
                "audio_object_key": conv.audio_object_key,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="清理 is_demo=0 的人工测试记录")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="待清理的 segment id，逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不执行删除")
    args = parser.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    if not ids:
        print("没有指定 id，退出")
        return 2

    session = SessionLocal()
    try:
        targets = _load_mysql(session, ids)

        if not targets:
            print(f"MySQL 中不存在这些 id：{ids}，无需清理")
            return 0

        missing = sorted(set(ids) - {t["segment_id"] for t in targets})
        if missing:
            print(f"[提示] 以下 id 在 MySQL 中已不存在：{missing}")

        # ---- 安全闸门 1：任何一条 is_demo=1 直接中止 ----
        demo_rows = [t for t in targets if t["is_demo"]]
        if demo_rows:
            print("[中止] 目标里含有 is_demo=1 的正式数据，拒绝删除：")
            for t in demo_rows:
                print(f"   seg={t['segment_id']}  {t['asr_text'][:30]}")
            return 3

        print(f"待清理 {len(targets)} 条（全部 is_demo=0）：")
        for t in targets:
            print(f"   seg={t['segment_id']}  conv={t['conversation_id']}  {t['start_time']}  {t['asr_text'][:36]}")

        # ---- 备份 ----
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = OUTPUT_DIR / f"test_records_backup_{stamp}.json"
        if not args.dry_run:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(
                json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n[备份] {backup_path}")

        if args.dry_run:
            print("\n[干跑] 未做任何修改。去掉 --dry-run 才会真正删除。")
            return 0

        # ---- 删 Milvus 向量 ----
        connections.connect(
            alias="default", host=settings.milvus_host, port=settings.milvus_port, timeout=10
        )
        if utility.has_collection(settings.milvus_collection):
            col = Collection(settings.milvus_collection)
            col.load()
            expr = "segment_id in [" + ", ".join(str(t["segment_id"]) for t in targets) + "]"
            before = col.query(expr=expr, output_fields=["segment_id"], limit=100)
            print(f"[Milvus] 删除前命中 {len(before)} 条向量，expr：{expr}")
            col.delete(expr)
            col.flush()
            after = col.query(expr=expr, output_fields=["segment_id"], limit=100)
            print(f"[Milvus] 删除后残留 {len(after)} 条")
        else:
            print("[Milvus] 集合不存在，跳过向量清理")

        # ---- 删 MySQL ----
        seg_ids = [t["segment_id"] for t in targets]
        conv_ids = [t["conversation_id"] for t in targets]
        deleted_seg = session.execute(
            text(f"DELETE FROM segments WHERE id IN ({','.join(str(i) for i in seg_ids)}) AND is_demo = 0")
        ).rowcount
        deleted_conv = session.execute(
            text(f"DELETE FROM conversations WHERE id IN ({','.join(str(i) for i in conv_ids)})")
        ).rowcount
        session.commit()
        print(f"\n[MySQL] 删除 segments {deleted_seg} 行，conversations {deleted_conv} 行")

        # ---- 复核 ----
        total_seg = session.execute(text("SELECT COUNT(*) FROM segments")).scalar()
        total_conv = session.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
        left_demo0 = session.execute(text("SELECT COUNT(*) FROM segments WHERE is_demo = 0")).scalar()
        print(f"[复核] 剩余 segments={total_seg}  conversations={total_conv}  is_demo=0 剩余={left_demo0}")

        # ---- 一致性：Milvus 里是否还有 MySQL 没有的孤儿向量 ----
        mysql_ids = {
            int(r[0]) for r in session.execute(text("SELECT id FROM segments")).fetchall()
        }
        if "col" in dir():
            all_vec = col.query(expr="segment_id >= 0", output_fields=["segment_id"], limit=16384)
            vec_ids = {int(v["segment_id"]) for v in all_vec}
            orphans = sorted(vec_ids - mysql_ids)
            print(f"[一致性] Milvus 向量 {len(vec_ids)} 条；MySQL 片段 {len(mysql_ids)} 条；孤儿向量 {len(orphans)} 条")
            if orphans:
                print(f"          孤儿 segment_id：{orphans[:20]}")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
