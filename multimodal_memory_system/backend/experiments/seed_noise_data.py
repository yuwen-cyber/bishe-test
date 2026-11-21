#!/usr/bin/env python3
"""把 1300 条噪声媒体灌入后端（走 POST /api/upload），形成 2000 条数据集。

要点：
- is_demo 一律传 true：避免被 search.py / hybrid.py 里「is_demo=0 优先」的排序规则顶到前排；
- duration_ms 取该条视频的实际切片时长（计划里的 duration_sec），贴近正式数据的 8~12 秒区间；
- 断点续传：mapping 里已有的 noise_id 自动跳过；
- 每条落盘一次 mapping，中断不丢进度。

用法（backend 目录，需后端 8000 已启动）：
    .venv\\Scripts\\python.exe experiments\\seed_noise_data.py --limit 30
    .venv\\Scripts\\python.exe experiments\\seed_noise_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
PLAN_PATH = NOISE_DIR / "noise_plan.json"
AUDIO_DIR = NOISE_DIR / "audio"
VIDEO_DIR = NOISE_DIR / "video"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
MAPPING_PATH = OUTPUT_DIR / "seed_noise_mapping.json"
FAILED_PATH = OUTPUT_DIR / "seed_noise_failed.json"


def _load_mapping() -> list[dict]:
    if MAPPING_PATH.exists():
        try:
            return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_mapping(mapping: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if args.start or args.limit:
        end = None if not args.limit else args.start + args.limit
        plan = plan[args.start:end]

    mapping = _load_mapping()
    done = {int(m["noise_id"]) for m in mapping}
    todo = [p for p in plan if int(p["noise_id"]) not in done]

    print(f"计划 {len(plan)} 条｜已灌 {len(done)} 条｜本次待灌 {len(todo)} 条")
    if not todo:
        return 0

    failed: list[dict] = []
    session = requests.Session()
    t0 = time.time()

    for idx, item in enumerate(todo, start=1):
        nid = int(item["noise_id"])
        a_path = AUDIO_DIR / f"noise_{nid:04d}.mp3"
        v_path = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        if not a_path.exists() or not v_path.exists():
            failed.append({"noise_id": nid, "error": "媒体文件缺失"})
            print(f"[{idx}/{len(todo)}] noise={nid} 缺文件，跳过", file=sys.stderr)
            continue

        duration_ms = int(round(float(item["duration_sec"]) * 1000))
        body = None
        last_err = ""
        for attempt in range(args.retry + 1):
            try:
                resp = session.post(
                    f"{args.base_url.rstrip('/')}/api/upload",
                    files={
                        "video": (v_path.name, v_path.read_bytes(), "video/mp4"),
                        "audio": (a_path.name, a_path.read_bytes(), "audio/mpeg"),
                    },
                    data={"duration_ms": str(duration_ms), "is_demo": "true"},
                    timeout=240,
                )
                resp.raise_for_status()
                body = resp.json()
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < args.retry:
                    time.sleep(2)

        if body is None:
            failed.append({"noise_id": nid, "group": item["group"], "error": last_err})
            print(f"[{idx}/{len(todo)}] noise={nid} FAILED: {last_err}", file=sys.stderr)
            continue

        mapping.append({
            "noise_id": nid,
            "group": item["group"],
            "labeled_emotion": item["emotion"],
            "db_segment_id": body.get("segment_id"),
            "conversation_id": body.get("conversation_id"),
            "backend_emotion": body.get("emotion"),
            "backend_asr_text": body.get("asr_text"),
            "duration_ms": duration_ms,
        })
        _save_mapping(mapping)

        el = time.time() - t0
        rate = idx / el * 60 if el > 0 else 0
        print(f"[{idx:>4}/{len(todo)}] noise={nid} db_seg={body.get('segment_id')} "
              f"标注={item['emotion']} 识别={body.get('emotion')} ({rate:.1f} 条/分)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_PATH.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    el = time.time() - t0
    print("\n========== 灌库结束 ==========")
    print(f"成功 {len(todo) - len(failed)} 条，失败 {len(failed)} 条，用时 {el:.0f}s")
    if todo:
        print(f"单条平均 {el / len(todo):.2f}s/条")
    print(f"映射 -> {MAPPING_PATH}")
    if failed:
        print(f"失败清单 -> {FAILED_PATH}（修复后重跑即可续传）")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
