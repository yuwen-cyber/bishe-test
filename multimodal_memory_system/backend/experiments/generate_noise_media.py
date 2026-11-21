#!/usr/bin/env python3
"""按 noise_plan.json 批量生成噪声媒体：TTS 配音 + 视频切片。

输出：
    experiments/noise/audio/noise_0000.mp3 ...
    experiments/noise/video/noise_0000.mp4 ...

断点续传：已存在的文件自动跳过；中断后重跑即可接着做。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\generate_noise_media.py --limit 30
    .venv\\Scripts\\python.exe experiments\\generate_noise_media.py          # 全量 1300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
PLAN_PATH = NOISE_DIR / "noise_plan.json"
AUDIO_DIR = NOISE_DIR / "audio"
VIDEO_DIR = NOISE_DIR / "video"

TTS_CONCURRENCY = 4


async def _tts_one(sem: asyncio.Semaphore, item: dict, out_path: Path) -> tuple[bool, float]:
    import edge_tts

    async with sem:
        t0 = time.time()
        try:
            tts = edge_tts.Communicate(
                text=item["text"],
                voice=item["voice"],
                rate="-5%",
            )
            await tts.save(str(out_path))
            return True, time.time() - t0
        except Exception as e:
            print(f"  [TTS失败] {item['noise_id']}: {type(e).__name__}: {e}", file=sys.stderr)
            return False, time.time() - t0


def _slice_video(item: dict, out_path: Path) -> tuple[bool, float]:
    t0 = time.time()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{item['start_sec']:.2f}",
        "-i", item["source_video"],
        "-t", f"{item['duration_sec']:.2f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-vf", "scale=640:-2",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, time.time() - t0
    except subprocess.CalledProcessError as e:
        print(f"  [切片失败] {item['noise_id']}: {e}", file=sys.stderr)
        return False, time.time() - t0


async def main_async(plan: list[dict]) -> int:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    todo = []
    for item in plan:
        nid = item["noise_id"]
        a = AUDIO_DIR / f"noise_{nid:04d}.mp3"
        v = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        if a.exists() and v.exists():
            continue
        todo.append(item)

    print(f"计划 {len(plan)} 条，已完成 {len(plan) - len(todo)} 条，本次待做 {len(todo)} 条")
    if not todo:
        return 0

    sem = asyncio.Semaphore(TTS_CONCURRENCY)
    t_all = time.time()
    tts_time = 0.0
    slice_time = 0.0
    ok = 0
    fail = 0

    # 先并发跑完这一批的 TTS，再串行切片（切片是 CPU 密集，串行更快）
    tts_tasks = []
    for item in todo:
        nid = item["noise_id"]
        a = AUDIO_DIR / f"noise_{nid:04d}.mp3"
        if a.exists():
            tts_tasks.append(asyncio.sleep(0, result=(True, 0.0)))
        else:
            tts_tasks.append(_tts_one(sem, item, a))

    tts_results = await asyncio.gather(*tts_tasks)
    tts_time = sum(r[1] for r in tts_results)

    for idx, (item, res) in enumerate(zip(todo, tts_results), start=1):
        nid = item["noise_id"]
        v = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        if not v.exists():
            ok_v, dt = _slice_video(item, v)
            slice_time += dt
            if not ok_v:
                fail += 1
                continue
        if not res[0]:
            fail += 1
            continue
        ok += 1
        if idx % 20 == 0 or idx == len(todo):
            el = time.time() - t_all
            rate = idx / el * 60 if el > 0 else 0
            print(f"[{idx:>4}/{len(todo)}] 用时 {el:.0f}s ({rate:.0f} 条/分)")

    total = time.time() - t_all
    print(f"\n完成：成功 {ok} 条，失败 {fail} 条")
    print(f"TTS 累计 {tts_time:.1f}s（并发{TTS_CONCURRENCY}）｜切片累计 {slice_time:.1f}s｜墙钟总计 {total:.1f}s")
    if ok:
        print(f"单条平均墙钟耗时：{total / len(todo):.2f}s/条")
        print(f"按 1300 条估算：约 {1300 * total / len(todo) / 3600:.2f} 小时")
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（试点用）")
    parser.add_argument("--start", type=int, default=0, help="从第 N 条开始")
    args = parser.parse_args()

    if not PLAN_PATH.exists():
        print(f"找不到 {PLAN_PATH}，先跑 build_noise_dataset.py")
        return 1

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if args.start or args.limit:
        end = None if not args.limit else args.start + args.limit
        plan = plan[args.start:end]

    return asyncio.run(main_async(plan))


if __name__ == "__main__":
    raise SystemExit(main())
