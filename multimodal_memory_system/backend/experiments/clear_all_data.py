#!/usr/bin/env python3
"""清空本系统的全部存量数据：MySQL（segments + conversations）、Milvus 集合、MinIO 对象。

【安全边界】
- 只操作本机 multimodal 数据库的 conversations / segments 两张表；
- 只删除 Milvus 中名为 settings.milvus_collection 的集合；
- 只删除 MinIO 中 settings.minio_bucket 桶内的对象；
- 不触碰其他数据库、其他桶，也不删除磁盘上的数据集文件。

用法（在 backend 目录下，使用后端 .venv 解释器）：
    .venv\\Scripts\\python.exe experiments\\clear_all_data.py --confirm

步骤建议：
    1. 先停掉正在运行的后端（避免旧连接持有缓存的 Milvus 句柄）；
    2. 运行本脚本清库；
    3. 重启后端（启动时会自动重建空的 Milvus 集合与表）；
    4. 再跑 seed_real_data.py 灌真实数据。

不带 --confirm 时只做“预检 + 打印将清理的数量”，不执行删除。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在 experiments/ 下运行时也能 import 到 backend 的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import models, milvus_client, storage  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal, engine  # noqa: E402
from sqlalchemy import text  # noqa: E402


def _count(db) -> tuple[int, int]:
    n_seg = db.query(models.Segment).count()
    n_conv = db.query(models.Conversation).count()
    return n_conv, n_seg


def main() -> int:
    parser = argparse.ArgumentParser(description="清空本系统存量数据（MySQL + Milvus + MinIO）")
    parser.add_argument("--confirm", action="store_true", help="真正执行删除；不加则只预检不删")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        n_conv, n_seg = _count(db)
        print(f"[预检] MySQL conversations={n_conv} 条, segments={n_seg} 条")
    finally:
        db.close()

    # MinIO 对象数量预检
    client = storage.get_client()
    objects = list(client.list_objects(settings.minio_bucket, recursive=True))
    print(f"[预检] MinIO 桶 {settings.minio_bucket} 内对象={len(objects)} 个")
    print(f"[预检] Milvus 集合={settings.milvus_collection}")

    if not args.confirm:
        print("\n未加 --confirm，以上仅为预检，未执行任何删除。")
        return 0

    # 1) MySQL：先删子表 segments，再删父表 conversations，最后重置自增
    db = SessionLocal()
    try:
        db.query(models.Segment).delete()
        db.query(models.Conversation).delete()
        db.commit()
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE segments AUTO_INCREMENT = 1"))
            conn.execute(text("ALTER TABLE conversations AUTO_INCREMENT = 1"))
        n_conv, n_seg = _count(db)
        print(f"[MySQL] 已清空；剩余 conversations={n_conv}, segments={n_seg}，自增已重置为 1")
    finally:
        db.close()

    # 2) MinIO：删除桶内全部对象
    removed = 0
    for obj in objects:
        client.remove_object(settings.minio_bucket, obj.object_name)
        removed += 1
    print(f"[MinIO] 已删除对象 {removed} 个")

    # 3) Milvus：整集合删除（下次后端启动时自动重建空集合）
    milvus_client.drop_collection()
    print(f"[Milvus] 集合 {settings.milvus_collection} 已删除，将在后端重启后自动重建")

    print("\n清库完成。请重启后端服务，再运行 seed_real_data.py 灌库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
