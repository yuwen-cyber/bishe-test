#!/usr/bin/env python3
"""用网上下载的无人物素材重新生成 1300 条噪声视频（替换原人脸切片）。

- 源：experiments/noise/sources/*.mp4（Mixkit Free License）
- 计划：experiments/noise/noise_plan.json（保留 noise_id / group / duration_sec）
- 输出：experiments/noise/video/noise_XXXX.mp4（覆盖）
- 映射：experiments/noise/noise_video_map.json（记录每条用的源视频与起点，供论文标注）

分配策略：素材按顺序逐条分配，尽量做到一源一条；分配完一轮再进入第二轮，
并把起点错开，避免同一源切出完全相同的画面。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\regenerate_noise_videos.py --limit 20
    .venv\\Scripts\\python.exe experiments\\regenerate_noise_videos.py
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
SOURCES_DIR = NOISE_DIR / "sources"
VIDEO_DIR = NOISE_DIR / "video"
PLAN = NOISE_DIR / "noise_plan.json"
MAP_OUT = NOISE_DIR / "noise_video_map.json"


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


def build_source_pool() -> list[dict]:
    pool = []
    for f in sorted(SOURCES_DIR.glob("*.mp4")):
        dur = probe_duration(f)
        if dur >= 2.0:
            pool.append({"path": str(f), "name": f.name, "duration": dur})
    return pool


def slice_one(src: dict, start: float, duration: float, out: Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-stream_loop", "-1",
        "-ss", f"{start:.2f}",
        "-i", src["path"],
        "-t", f"{duration:.2f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-vf", "scale=640:-2",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not PLAN.exists():
        print(f"缺少 {PLAN}")
        return 1

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    pool = build_source_pool()
    if not pool:
        print("sources 目录里没有可用素材")
        return 1
    print(f"计划 {len(plan)} 条｜可用素材 {len(pool)} 个｜素材总时长 "
          f"{sum(s['duration'] for s in pool)/60:.1f} 分钟")

    rng = random.Random(args.seed)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if args.start or args.limit:
        end = None if not args.limit else args.start + args.limit
        todo = plan[args.start:end]
    else:
        todo = plan

    # 断点续传：记录文件都在才跳过
    existing_map: dict[str, dict] = {}
    if MAP_OUT.exists():
        try:
            existing_map = {str(x["noise_id"]): x
                            for x in json.loads(MAP_OUT.read_text(encoding="utf-8"))}
        except Exception:
            existing_map = {}

    mapping = dict(existing_map)
    ok = fail = skip = 0
    t0 = time.time()

    for idx, item in enumerate(todo, 1):
        nid = item["noise_id"]
        out = VIDEO_DIR / f"noise_{nid:04d}.mp4"
        dur = float(item["duration_sec"])
        if str(nid) in mapping and out.exists() and out.stat().st_size > 100_000:
            skip += 1
            continue

        # 一轮分配：每轮使用全部素材，轮次决定起点偏置
        use_idx = nid % len(pool)
        round_no = nid // len(pool)
        src = pool[use_idx]
        max_start = max(0.2, src["duration"] - dur - 0.3)
        if max_start <= 0.3:
            start = 0.0
        else:
            # 每多轮一次，起点基准向后退，减少同源同画面
            base = (round_no * 0.35 + rng.random() * 0.5) % 1.0
            start = min(max_start, base * max_start)

        ok_one = slice_one(src, start, dur, out)
        if ok_one:
            ok += 1
            mapping[str(nid)] = {
                "noise_id": nid,
                "group": item.get("group"),
                "source": src["name"],
                "source_duration": round(src["duration"], 2),
                "start_sec": round(start, 2),
                "duration_sec": dur,
            }
        else:
            fail += 1

        if idx % 40 == 0 or idx == len(todo):
            el = time.time() - t0
            done = idx - skip
            rate = done / el * 60 if el > 0 else 0
            print(f"[{idx}/{len(todo)}] 成功 {ok} 失败 {fail} 跳过 {skip}｜{rate:.0f} 条/分")

    MAP_OUT.write_text(
        json.dumps([mapping[k] for k in sorted(mapping, key=lambda x: int(x))],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成：成功 {ok}，失败 {fail}，跳过 {skip}｜用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"映射写入 {MAP_OUT}")

    # 复用统计
    files = list(VIDEO_DIR.glob("noise_*.mp4"))
    used = {mapping[str(int(f.stem.split('_')[1]))]["source"]
            for f in files if str(int(f.stem.split('_')[1])) in mapping}
    print(f"当前噪声视频文件 {len(files)} 个，来自 {len(used)} 个不同源素材")
    if files:
        cnt: dict[str, int] = {}
        for f in files:
            key = str(int(f.stem.split('_')[1]))
            if key in mapping:
                cnt[mapping[key]["source"]] = cnt.get(mapping[key]["source"], 0) + 1
        if cnt:
            mx = max(cnt.items(), key=lambda kv: kv[1])
            print(f"单源最多复用 {mx[1]} 次（{mx[0]}），平均 {sum(cnt.values())/len(cnt):.1f} 次")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
