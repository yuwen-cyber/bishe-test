#!/usr/bin/env python3
"""为正式 700 条数据生成"情绪匹配"的真实人脸切片，替换原来的统一占位视频。

背景：原 700 条的 video_xxx.mp4 是同一个 placeholder.mp4 的复制件，导致 CLIP 图像向量
700 条完全相同，图像模态形同摆设。本脚本按每条记录的 emotion 从 face_videos 里挑选
同名情绪的真实片段，随机起点截取与 duration_ms 等长的一段（素材不够长时循环补齐）。

输出：experiments/real_videos/seg_XXXX.mp4（640 宽、无音轨、crf30）
用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\generate_real_slices.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXPERIMENTS_DIR.parent
FACE_DIR = Path(r"D:\biyesheji\face_videos")
OUT_DIR = EXPERIMENTS_DIR / "real_videos"
DATASET = EXPERIMENTS_DIR / "segments_dataset.json"

EMO_PREFIX = {
    "喜悦": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "scared",
    "惊讶": "surprised",
    "厌恶": "disgusted",
    "中性": "expressionless",
}


def probe(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def build_pools() -> dict[str, list[tuple[Path, float]]]:
    pools: dict[str, list[tuple[Path, float]]] = {v: [] for v in EMO_PREFIX.values()}
    for f in FACE_DIR.glob("*.mp4"):
        name = f.stem
        for prefix in EMO_PREFIX.values():
            if name.startswith(prefix):
                dur = probe(f)
                if dur > 1.0:
                    pools[prefix].append((f, dur))
                break
    return pools


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    pools = build_pools()
    for k, v in pools.items():
        print(f"素材池 {k}: {len(v)} 段")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    ok = skip = fail = 0

    for item in dataset:
        sid = int(item["segment_id"])
        target = max(3.0, float(item.get("duration_ms", 15000)) / 1000.0)
        prefix = EMO_PREFIX.get(item.get("emotion") or "中性")
        pool = pools.get(prefix) or []
        if not pool:
            fail += 1
            print(f"seg {sid}: 情绪 {item.get('emotion')} 无素材", file=sys.stderr)
            continue

        out_path = OUT_DIR / f"seg_{sid:04d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            skip += 1
            continue

        src, src_dur = random.choice(pool)
        # 起点随机，但保留 1 秒余量；素材不足 target 时循环
        start = round(random.uniform(0, max(0.0, src_dur - 1.0 - min(target, src_dur - 0.5))), 3)
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{start:.3f}", "-i", str(src),
               "-t", f"{target:.3f}", "-an",
               "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
               str(out_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            ok += 1
            if ok % 50 == 0:
                print(f"已生成 {ok} 条（最近 seg {sid}，源 {src.name}）", flush=True)
        except Exception as e:
            fail += 1
            print(f"seg {sid} 生成失败: {e}", file=sys.stderr)

    print(f"\n完成：新生成 {ok} 条，已存在跳过 {skip} 条，失败 {fail} 条")
    print(f"输出目录：{OUT_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
