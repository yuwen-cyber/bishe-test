#!/usr/bin/env python3
"""批量下载噪声素材（Mixkit 360p），带断点续传与坏文件校验。

输出：experiments/noise/sources/{vid}.mp4
清单：experiments/noise/source_plan.json
校验：下载后用 ffprobe 检查时长，<2 秒或无法解析的删除

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\download_noise_sources.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
PLAN = NOISE_DIR / "source_plan.json"
OUT_DIR = NOISE_DIR / "sources"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def fetch_one(item: dict) -> tuple[str, str]:
    dst = OUT_DIR / f"{item['vid']}.mp4"
    if dst.exists() and probe_duration(dst) >= 2.0:
        return item["vid"], "skip"

    tmp = dst.with_suffix(".part")
    try:
        r = requests.get(item["url"], headers={"User-Agent": UA}, timeout=60, stream=True)
        if r.status_code != 200:
            return item["vid"], f"http{r.status_code}"
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        if probe_duration(tmp) < 2.0:
            tmp.unlink(missing_ok=True)
            return item["vid"], "bad"
        tmp.replace(dst)
        return item["vid"], "ok"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return item["vid"], f"err:{type(e).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not PLAN.exists():
        print(f"缺少 {PLAN}，先跑 fetch_noise_sources.py")
        return 1

    items = json.loads(PLAN.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = {p.stem for p in OUT_DIR.glob("*.mp4") if probe_duration(p) >= 2.0}
    todo = [x for x in items if x["vid"] not in done]
    print(f"清单 {len(items)}，已有可用 {len(done)}，本次待下 {len(todo)}")

    stats: dict[str, int] = {}
    total_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, x): x for x in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            vid, status = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if status in ("ok", "skip"):
                total_ok += 1
            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] 成功 {total_ok}｜状态 {stats}")

    files = list(OUT_DIR.glob("*.mp4"))
    total_dur = sum(probe_duration(p) for p in files)
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    print(f"\n完成：可用素材 {len(files)} 个｜{total_mb:.0f} MB｜总时长 {total_dur/60:.1f} 分钟")
    print(f"平均每条 {total_dur/max(len(files),1):.1f} 秒；1300 条 ×10 秒需 {13000/60:.0f} 分钟素材")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
