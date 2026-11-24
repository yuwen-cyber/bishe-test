#!/usr/bin/env python3
"""生成可上传网盘的完整备份包。

包含：
    01_minio_media.zip      MinIO 里的音视频原件（不可再生，优先级最高）
    02_experiments_media.zip  real_videos / noise 全部素材
    03_database.zip         两张表的全量数据（JSON，含恢复脚本）
    04_config_docs.zip       .env、README、恢复说明
    SHA256SUMS.txt          各包校验值

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\make_backup.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from minio import Minio  # noqa: E402

from app import models  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402

EXPERIMENTS = BACKEND / "experiments"
STAMP = datetime.now().strftime("%Y-%m-%d")
OUT = Path("D:/biyesheji") / f"_backup_{STAMP}"


def zip_dir(src: Path, dst: Path, arc_root: str) -> int:
    """ZIP_STORED（视频不压缩，速度快），返回写入文件数。"""
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f"{arc_root}/{f.relative_to(src).as_posix()}")
                n += 1
    return n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_minio(dst_zip: Path) -> int:
    client = Minio(
        settings.minio_endpoint,
        settings.minio_access_key,
        settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    tmp = OUT / "_minio_tmp"
    n = 0
    for obj in client.list_objects(settings.minio_bucket, recursive=True):
        if obj.is_dir:
            continue
        dst = tmp / obj.object_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(settings.minio_bucket, obj.object_name, str(dst))
        n += 1
        if n % 200 == 0:
            print(f"  MinIO 已导出 {n} 个对象...", flush=True)
    zip_dir(tmp, dst_zip, "minio")
    shutil.rmtree(tmp)
    return n


def export_db(dst_dir: Path) -> dict:
    db = SessionLocal()
    data = {}
    try:
        for table in (models.Conversation, models.Segment):
            name = table.__tablename__
            cols = [c.name for c in table.__table__.columns]
            rows = []
            for obj in db.query(table).all():
                rows.append({c: getattr(obj, c) for c in cols})
            data[name] = {"columns": cols, "rows": rows}
    finally:
        db.close()
    (dst_dir / "database.json").write_text(
        json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {k: len(v["rows"]) for k, v in data.items()}


RESTORE_DOC = """# 恢复说明

## 备份内容

| 文件 | 内容 |
|---|---|
| `01_minio_media.zip` | MinIO 里的音视频原件（**不可再生，最重要**） |
| `02_experiments_media.zip` | real_videos / noise 全部素材 |
| `03_database.zip` | MySQL 两张表的全量数据（database.json） |
| `04_config_docs.zip` | .env（含真实密码）、README、答辩手册 |
| `SHA256SUMS.txt` | 各包校验值 |

## 恢复步骤

1. 校验完整性（Windows PowerShell）：
   ```powershell
   Get-FileHash *.zip -Algorithm SHA256
   ```
   与 `SHA256SUMS.txt` 逐行比对，不一致就重新下载。

2. 还原代码：
   ```bash
   git clone https://github.com/yuwen-cyber/bishe-test.git D:\biyesheji
   ```
   （公开仓库含全部源码与实验脚本）

3. 还原素材：把 `02_experiments_media.zip` 解压到
   `multimodal_memory_system/backend/experiments/`，覆盖同名目录。

4. 还原配置：把 `04_config_docs.zip` 里的 `.env` 放到
   `multimodal_memory_system/backend/.env`。

5. 还原数据库：
   ```bash
   # 先建库
   mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS multimodal"
   # 再用脚本导入（需要 backend 的 venv 与依赖）
   .venv\\Scripts\\python.exe experiments\\restore_db.py
   ```
   `restore_db.py` 会读取 `03_database.zip` 解压出的 `database.json` 写回两张表。

6. 还原 MinIO：
   ```bash
   # 启动 MinIO 后，用 backend venv 跑
   .venv\\Scripts\\python.exe experiments\\restore_minio.py --src <解压出的 minio 目录>
   ```

7. 重建向量库：Milvus 数据可重算，见 `experiments/` 下的导入脚本；
   也可从 `restore_milvus.py` 按需重放。

## 注意

- `.env` 含真实数据库密码与 API key，网盘务必设为私有，不要分享链接。
- 优先保证 `01_minio_media.zip` 上传成功：700 条真实录音只存在这里。
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{OUT}")

    # 01 MinIO
    print("[1/4] 导出 MinIO 音视频原件 ...")
    n_minio = export_minio(OUT / "01_minio_media.zip")
    print(f"  完成，{n_minio} 个对象")

    # 02 实验素材
    print("[2/4] 打包 experiments 素材（real_videos / noise）...")
    tmp_media = OUT / "_media_tmp" / "experiments"
    tmp_media.mkdir(parents=True, exist_ok=True)
    for sub in ("real_videos", "noise"):
        src = EXPERIMENTS / sub
        if src.exists():
            dst = tmp_media / sub
            print(f"  复制 {sub} ...", flush=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
    zip_dir(tmp_media, OUT / "02_experiments_media.zip", "experiments")
    shutil.rmtree(OUT / "_media_tmp", ignore_errors=True)

    # 03 数据库
    print("[3/4] 导出数据库 ...")
    db_dir = OUT / "_db_tmp"
    db_dir.mkdir(exist_ok=True)
    counts = export_db(db_dir)
    (db_dir / "restore_db.py").write_text(
        '''import json
from pathlib import Path

from app import models
from app.database import SessionLocal, engine

models.Base.metadata.create_all(engine)
data = json.loads((Path(__file__).parent / "database.json").read_text(encoding="utf-8"))
db = SessionLocal()
try:
    for name, payload in data.items():
        model = models.Conversation if name == "conversations" else models.Segment
        db.query(model).delete()
        for row in payload["rows"]:
            db.add(model(**{k: (None if v is None else v) for k, v in row.items()}))
    db.commit()
    print("数据库已恢复")
finally:
    db.close()
''',
        encoding="utf-8",
    )
    zip_dir(db_dir, OUT / "03_database.zip", "database")
    shutil.rmtree(db_dir)
    print(f"  conversations={counts.get('conversations')}，segments={counts.get('segments')}")

    # 04 配置与文档
    print("[4/4] 打包配置与文档 ...")
    doc_dir = OUT / "_doc_tmp"
    doc_dir.mkdir(exist_ok=True)
    for f, arc in (
        (BACKEND / ".env", "backend/.env"),
        (BACKEND / ".env.example", "backend/.env.example"),
        (Path("D:/biyesheji/README.md"), "README.md"),
        (Path("D:/biyesheji/答辩演示操作手册.md"), "答辩演示操作手册.md"),
    ):
        if f.exists():
            target = doc_dir / arc
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    (doc_dir / "恢复说明.md").write_text(RESTORE_DOC, encoding="utf-8")
    zip_dir(doc_dir, OUT / "04_config_docs.zip", "")
    shutil.rmtree(doc_dir)

    # SHA256
    lines = []
    total = 0
    for z in sorted(OUT.glob("*.zip")):
        lines.append(f"{sha256(z)}  {z.name}")
        total += z.stat().st_size
        print(f"  {z.name}: {z.stat().st_size/1024/1024:.1f} MB")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "恢复说明.md").write_text(RESTORE_DOC, encoding="utf-8")

    print(f"\n备份完成：{len(lines)} 个包，共 {total/1024/1024:.1f} MB")
    print(f"目录：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
