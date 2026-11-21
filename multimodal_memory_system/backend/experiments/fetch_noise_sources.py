#!/usr/bin/env python3
"""为噪声数据抓取"无人物"免费素材视频清单（Mixkit Free License，可免费使用）。

用途：把 1300 条噪声视频从"复用人脸视频"换成风景/城市/物件等场景切片，
避免与正式 700 条人脸素材重复。

输出：experiments/noise/source_plan.json
    [{"category": "...", "vid": "...", "url": "...", "page": N}]

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\fetch_noise_sources.py --pages 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
OUT = NOISE_DIR / "source_plan.json"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

# 只取无人物/无明显人脸的分类，避免与正式 700 条的人脸素材重合
CATEGORIES = [
    "nature", "city", "food", "abstract",
    "technology", "animals", "business", "sport",
]

PAT_STD = re.compile(r"assets\.mixkit\.co/videos/(\d+)/\1-360\.mp4")
PAT_ACTIVE = re.compile(
    r"assets\.mixkit\.co/active_storage/video_items/(\d+)/(\d+)/\1-video-360\.mp4"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=5, help="每个分类抓多少页")
    args = parser.parse_args()

    found: dict[str, dict] = {}
    for cat in CATEGORIES:
        for page in range(1, args.pages + 1):
            url = f"https://mixkit.co/free-stock-video/{cat}/?page={page}"
            try:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"[跳过] {cat} p{page}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            hits = 0
            for m in PAT_STD.finditer(r.text):
                vid = m.group(1)
                found.setdefault(vid, {
                    "vid": vid, "category": cat, "page": page,
                    "url": f"https://assets.mixkit.co/videos/{vid}/{vid}-360.mp4",
                    "source_page": url,
                })
                hits += 1
            for m in PAT_ACTIVE.finditer(r.text):
                vid, stamp = m.group(1), m.group(2)
                key = "a" + vid
                found.setdefault(key, {
                    "vid": key, "category": cat, "page": page,
                    "url": f"https://assets.mixkit.co/active_storage/video_items/{vid}/{stamp}/{vid}-video-360.mp4",
                    "source_page": url,
                })
                hits += 1
            print(f"{cat} p{page}: {hits} 条链接，累计去重 {len(found)}")

    items = list(found.values())
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n去重后共 {len(items)} 个候选视频 -> {OUT}")
    by_cat: dict[str, int] = {}
    for x in items:
        by_cat[x["category"]] = by_cat.get(x["category"], 0) + 1
    print("按分类：", by_cat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
