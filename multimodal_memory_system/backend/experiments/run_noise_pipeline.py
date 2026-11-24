#!/usr/bin/env python3
"""噪声数据自动化流水线：抓取 → 下载 → 切片 → 入库重建 → 校验。

把原来手动分步执行的 5 个脚本串成一条命令。各阶段脚本本身支持断点续传，
中断后重跑会跳过已完成部分，因此本脚本可以反复执行。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\run_noise_pipeline.py
    .venv\\Scripts\\python.exe experiments\\run_noise_pipeline.py --stages fetch,download
    .venv\\Scripts\\python.exe experiments\\run_noise_pipeline.py --from regenerate
    .venv\\Scripts\\python.exe experiments\\run_noise_pipeline.py --skip fetch,download
    .venv\\Scripts\\python.exe experiments\\run_noise_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
NOISE_DIR = EXPERIMENTS_DIR / "noise"

# (阶段名, 脚本, 默认参数, 说明)
STAGES = [
    ("fetch", "fetch_noise_sources.py", ["--pages", "5"],
     "抓取素材清单 → noise/source_plan.json"),
    ("download", "download_noise_sources.py", ["--workers", "8"],
     "下载素材 → noise/sources/*.mp4（断点续传 + ffprobe 校验）"),
    ("regenerate", "regenerate_noise_videos.py", [],
     "切片生成 1300 条噪声视频 → noise/video/*.mp4 + noise_video_map.json"),
    ("replace", "replace_noise_video_and_rebuild.py", [],
     "覆盖 MinIO 同 key 对象并重算 CLIP 图像向量、重写 Milvus 行"),
    ("verify", "verify_noise_replace.py", [],
     "校验 Milvus 行数 / 重复 id / 抽样向量 / MinIO 对象大小"),
]

STAGE_NAMES = [s[0] for s in STAGES]

# 这些阶段需要 MinIO(9000) 与 Milvus(19530) 在线
SERVICE_STAGES = {"replace", "verify"}
PORTS = [(9000, "MinIO"), (19530, "Milvus")]


def check_port(port: int, name: str) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def wait_for_milvus(max_wait: int = 150) -> bool:
    """端口通不代表可查询：刚启动时 QueryNode 时间戳未追上，会报 Timestamp lag。

    这里等 Milvus 真正能返回一行数据再放行，避免 replace/verify 白跑一趟。
    """
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    try:
        from pymilvus import Collection, connections
        from app.config import settings
    except Exception as e:
        print(f"[前置检查] 无法加载 pymilvus 或 app 配置，跳过就绪探测：{e}")
        return True

    t0 = time.time()
    alias = "pipeline_probe"
    last_err = ""
    while time.time() - t0 < max_wait:
        try:
            connections.connect(alias=alias, host=settings.milvus_host,
                                port=settings.milvus_port)
            col = Collection(settings.milvus_collection, using=alias)
            col.load()
            col.query(expr="segment_id >= 0", output_fields=["segment_id"], limit=1)
            print(f"[前置检查] Milvus 已可查询（等待 {time.time() - t0:.0f}s）")
            return True
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            try:
                connections.disconnect(alias)
            except Exception:
                pass
            time.sleep(8)

    print(f"[前置检查] Milvus {max_wait}s 内仍不可查询，最后错误：{last_err}")
    print("通常是刚启动、QueryNode 尚未追上时间戳，等 1-2 分钟后重跑本命令即可")
    return False


def preflight(stages: list[str]) -> bool:
    """跑 replace / verify 前确认依赖服务在线，避免跑一半报 WinError 10061。"""
    if not SERVICE_STAGES.intersection(stages):
        return True

    missing = [f"{name}(:{port})" for port, name in PORTS if not check_port(port, name)]
    if missing:
        print("\n[前置检查失败] 以下服务未启动：" + "、".join(missing))
        print("请先执行：")
        print('  Start-Process "D:\\DockerDesktop\\Docker Desktop.exe"')
        print("  docker start mm-etcd mm-minio")
        print("  等待约 6 秒")
        print("  docker start mm-milvus")
        return False

    if not wait_for_milvus():
        return False

    # 检查上游产物是否齐全
    if "replace" in stages:
        plan = NOISE_DIR / "noise_plan.json"
        vdir = NOISE_DIR / "video"
        mapping = OUTPUT_DIR / "batch_import_mapping.json"
        if not plan.exists():
            print(f"\n[前置检查失败] 缺少 {plan}，请先跑 fetch/regenerate 阶段")
            return False
        if not mapping.exists():
            print(f"\n[前置检查失败] 缺少 {mapping}，噪声数据尚未入库")
            return False
        vids = list(vdir.glob("noise_*.mp4")) if vdir.exists() else []
        if not vids:
            print(f"\n[前置检查失败] {vdir} 下没有切片，请先跑 regenerate 阶段")
            return False
        print(f"[前置检查通过] 服务在线；切片 {len(vids)} 个；映射 {mapping.name} 存在")
    return True


def resolve_stages(args) -> list[str]:
    """按 --stages / --skip / --from 三个开关算出实际要执行的阶段顺序。"""
    order = list(STAGE_NAMES)

    if args.stages:
        picked = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in picked if s not in STAGE_NAMES]
        if unknown:
            raise SystemExit(f"未知阶段：{unknown}，可选：{STAGE_NAMES}")
        order = [s for s in STAGE_NAMES if s in picked]
    elif args.start_from:
        if args.start_from not in STAGE_NAMES:
            raise SystemExit(f"未知阶段：{args.start_from}，可选：{STAGE_NAMES}")
        order = order[order.index(args.start_from):]

    if args.skip:
        skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
        order = [s for s in order if s not in skipped]

    if not order:
        raise SystemExit("没有需要执行的阶段（被 --skip 全部排除）")
    return order


def run_stage(name: str, script: str, extra: list[str], args, log) -> int:
    params = list(extra)
    if name == "fetch" and args.pages:
        params = ["--pages", str(args.pages)]
    if name == "download" and args.workers:
        params = ["--workers", str(args.workers)]
    if name == "regenerate" and args.limit:
        params = ["--limit", str(args.limit)]
    if name == "replace" and args.limit:
        params = ["--limit", str(args.limit)]

    cmd = [sys.executable, str(EXPERIMENTS_DIR / script)] + params
    print("\n" + "=" * 68)
    print(f"[{name}] {script} {' '.join(params)}".rstrip())
    print("=" * 68)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    rc = proc.wait()
    elapsed = time.time() - t0

    if rc == 0:
        print(f"[{name}] 完成，耗时 {elapsed:.1f}s")
    else:
        print(f"[{name}] 失败，返回码 {rc}，耗时 {elapsed:.1f}s")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="噪声数据自动化流水线（抓取/下载/切片/入库/校验）")
    parser.add_argument("--stages", help="只跑指定阶段，逗号分隔，如 fetch,download")
    parser.add_argument("--skip", help="跳过指定阶段，逗号分隔")
    parser.add_argument("--from", dest="start_from", help="从指定阶段开始，跑完后续全部")
    parser.add_argument("--pages", type=int, help="覆盖 fetch 的每类页数（默认 5）")
    parser.add_argument("--workers", type=int, help="覆盖 download 的并发数（默认 8）")
    parser.add_argument("--limit", type=int, help="覆盖 regenerate/replace 的条数上限")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = parser.parse_args()

    stages = resolve_stages(args)

    # 按阶段定义取脚本与说明
    table = {s[0]: s for s in STAGES}
    print("执行计划：")
    for i, name in enumerate(stages, 1):
        _, script, params, desc = table[name]
        print(f"  {i}. {name:<11} {script:<38} {desc}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何阶段")
        return 0

    if not preflight(stages):
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"noise_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"\n日志：{log_path}")

    t_all = time.time()
    failed_at = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"pipeline start {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log.write(f"stages: {','.join(stages)}\n")
        for name in stages:
            _, script, params, _ = table[name]
            rc = run_stage(name, script, params, args, log)
            if rc != 0:
                failed_at = name
                log.write(f"\nFAILED at stage {name} rc={rc}\n")
                break

    total = time.time() - t_all
    print("\n" + "-" * 68)
    if failed_at:
        print(f"流水线中断：{failed_at} 阶段失败，总耗时 {total:.1f}s")
        print("排查后可重跑同一命令，已完成的阶段会自动跳过或续传。")
        return 1

    print(f"流水线全部完成，总耗时 {total:.1f}s（{total/60:.1f} 分钟）")
    print("下一步建议：重跑评测并导出演示数据")
    print("  .venv\\Scripts\\python.exe experiments\\run_ablation.py")
    print("  .venv\\Scripts\\python.exe experiments\\run_fusion_ablation.py")
    print("  .venv\\Scripts\\python.exe experiments\\run_eval.py")
    print("  .venv\\Scripts\\python.exe experiments\\run_hybrid_eval.py")
    print("  .venv\\Scripts\\python.exe experiments\\export_demo_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
