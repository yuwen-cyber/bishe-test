#!/usr/bin/env python3
"""抽取噪声视频中间帧，用 ffmpeg xstack 拼成 4×2 预览图（无需 PIL）。

输出：D:\\biyesheji\\噪声素材预览.png
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
NOISE = BACKEND / "experiments" / "noise"
VIDEO_DIR = NOISE / "video"
MAP = json.loads((NOISE / "noise_video_map.json").read_text(encoding="utf-8"))
PLAN = json.loads((NOISE / "source_plan.json").read_text(encoding="utf-8"))
CAT = {x["vid"] + ".mp4": x["category"] for x in PLAN}

rng = random.Random(7)
by_cat: dict[str, list[dict]] = {}
for m in MAP:
    by_cat.setdefault(CAT.get(m["source"], "unknown"), []).append(m)

order = ["nature", "city", "food", "technology", "animals", "abstract", "business", "sport"]
picked = [rng.choice(by_cat[c]) for c in order if by_cat.get(c)][:8]

with tempfile.TemporaryDirectory() as td:
    frames = []
    for i, m in enumerate(picked):
        png = Path(td) / f"f{i}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(m['start_sec']) + 1.0:.2f}",
             "-i", str(VIDEO_DIR / f"noise_{int(m['noise_id']):04d}.mp4"),
             "-frames:v", "1", "-vf", "scale=640:360", str(png)],
            check=True, capture_output=True,
        )
        frames.append(png)

    # 4 列 × 2 行
    inputs = []
    for f in frames:
        inputs += ["-i", str(f)]
    filt = (
        "[0:v][1:v][2:v][3:v]hstack=inputs=4[r1];"
        "[4:v][5:v][6:v][7:v]hstack=inputs=4[r2];"
        "[r1][r2]vstack"
    )
    out = Path(r"D:\biyesheji\噪声素材预览.png")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", filt, str(out)],
        check=True, capture_output=True,
    )

print(f"已生成 {out}")
for m in picked:
    print(f"  noise_{int(m['noise_id']):04d}  分类={CAT.get(m['source'], '?')}  源={m['source']}")
