#!/usr/bin/env python3
"""从本地长视频（影片/纪录片）自动切出 8~12 秒无面部片段，作为噪声数据。

与下载网络短素材不同，本脚本面对的是本地已有的**长视频文件**，因此真实实现了
"长视频 → 短视频"的自动切分环节：

    1. ffmpeg 按指定帧率解码，经管道直接送入内存（不落盘）；
    2. Haar 级联逐帧检测人脸（1280x720 高分辨率，避免小脸漏检），记录含面部时间点；
    3. 时间点前后各扩 2 秒并合并成"禁区"，其余为可切区间；
    4. 可切区间内按固定窗口切片（默认 10 秒，窗口间留 0.5 秒间隔）；
    5. 多源时按轮询方式取样，避免切片集中在单个影片；
    6. **切片后逐条复核**：对每条切片再抽 6 帧做 1280x720 检测，检出人脸即丢弃，
       并从候选池补齐，直到达到目标条数；
    7. 导出 640 宽、无音轨、crf 30 的切片，并写映射文件。

排除含面部片段的原因：面部表情本身携带情绪信号，若混入噪声，会使"噪声语义中性"
这一实验前提失效（详见论文数据集章节的载体选择说明）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\build_film_slices.py --dir "C:\\影片" --probe
    .venv\\Scripts\\python.exe experiments\\build_film_slices.py --dir "C:\\影片" --write --target 300 --audit
    .venv\\Scripts\\python.exe experiments\\build_film_slices.py --sources a.mp4 b.mp4 --write
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
FILM_DIR = NOISE_DIR / "film"
MAP_OUT = NOISE_DIR / "film_slice_map.json"
CANDIDATES_OUT = NOISE_DIR / "film_candidates.json"

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".rmvb", ".wmv", ".flv", ".ts", ".m4v"}

# 扫描用分辨率：480x270 实测只能检出 1/20 帧，1280x720 可检出 19/20
SCAN_W, SCAN_H = 1280, 720
AUDIT_SAMPLES = 6


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


def _stream_frames(src: Path, fps: float, w: int, h: int):
    """按帧流式输出 BGR 数组。"""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(src),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    finally:
        proc.stdout.close()
        proc.wait()


def detect_face_times(src: Path, fps: float) -> tuple[list[float], int]:
    """扫描全片，返回（含面部的时间点秒，总帧数）。"""
    hits: list[float] = []
    n = 0
    for img in _stream_frames(src, fps, SCAN_W, SCAN_H):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces):
            hits.append(n / fps)
        n += 1
    return hits, n


def audit_clip(path: Path, samples: int = AUDIT_SAMPLES) -> int:
    """对切片抽若干帧复核，返回检出人脸的帧数（0 表示干净）。"""
    dur = probe_duration(path) or 10.0
    fps = max(0.5, min(2.0, samples / max(dur, 1.0)))
    bad = 0
    for img in _stream_frames(path, fps, SCAN_W, SCAN_H):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))):
            bad += 1
    return bad


def to_intervals(hits: list[float], total: float, pad: float = 2.0) -> list[list[float]]:
    iv: list[list[float]] = []
    for t in sorted(hits):
        a, b = max(0.0, t - pad), min(total, t + pad)
        if iv and a <= iv[-1][1]:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    return iv


def free_segments(blocked: list[list[float]], total: float) -> list[tuple[float, float]]:
    free: list[tuple[float, float]] = []
    cur = 0.0
    for a, b in blocked:
        if a - cur > 0.05:
            free.append((cur, a))
        cur = max(cur, b)
    if total - cur > 0.05:
        free.append((cur, total))
    return free


def plan_slices(free: list[tuple[float, float]], window: float,
                min_len: float, gap: float = 0.5) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in free:
        span = b - a
        if span < min_len:
            continue
        if span < window:
            out.append((a, span))
            continue
        t = a
        while t + window <= b + 1e-6:
            out.append((t, window))
            t += window + gap
    return out


def take_round_robin(plan: list[dict], target: int) -> list[dict]:
    if target <= 0 or len(plan) <= target:
        return plan
    by_src: dict[str, list[dict]] = collections.defaultdict(list)
    for p in plan:
        by_src[p["source"]].append(p)
    out: list[dict] = []
    idx = 0
    while len(out) < target:
        added = False
        for key in sorted(by_src):
            if idx < len(by_src[key]) and len(out) < target:
                out.append(by_src[key][idx])
                added = True
        if not added:
            break
        idx += 1
    return out


def cut_clip(src: Path, start: float, dur: float, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-an", "-vf", "scale=640:-2", "-c:v", "libx264", "-crf", "30",
         "-pix_fmt", "yuv420p", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def collect_sources(args) -> list[Path]:
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            raise SystemExit(f"目录不存在：{d}")
        found = sorted(p for p in d.rglob("*")
                       if p.is_file() and p.suffix.lower() in VIDEO_EXT)
        if not found:
            raise SystemExit(f"目录下没有视频文件：{d}")
        return found
    return [Path(p) for p in (args.sources or [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="长视频自动切成无面部短视频")
    parser.add_argument("--dir", help="影片目录，递归扫描视频文件")
    parser.add_argument("--sources", nargs="*", help="直接指定影片路径")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--min-len", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=0.5, help="扫描抽帧频率（默认 0.5）")
    parser.add_argument("--target", type=int, default=0, help="目标条数（0 表示全部）")
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--audit", action="store_true",
                        help="切片后逐条复核，检出人脸则丢弃并补片")
    parser.add_argument("--probe", action="store_true", help="只统计，不切片")
    parser.add_argument("--write", action="store_true", help="切片并写映射")
    args = parser.parse_args()

    sources = collect_sources(args)
    if not sources:
        raise SystemExit("请用 --dir 或 --sources 指定影片")
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        print("找不到影片：")
        for m in missing:
            print("  " + m)
        return 2

    print(f"共 {len(sources)} 个源｜窗口 {args.window:.0f}s｜扫描 {SCAN_W}x{SCAN_H}"
          f" @ {args.fps}fps｜目标 {args.target or '不限'}｜复核 {'开' if args.audit else '关'}\n")

    all_plan: list[dict] = []
    src_stats: list[dict] = []
    t_start = time.time()

    for idx, src in enumerate(sources, 1):
        total = probe_duration(src)
        if total <= 0:
            print(f"[{idx}/{len(sources)}] 跳过（无法读取时长）：{src.name}")
            continue

        t0 = time.time()
        hits, n_scan = detect_face_times(src, args.fps)
        blocked = to_intervals(hits, total)
        free = free_segments(blocked, total)
        slices = plan_slices(free, args.window, args.min_len)
        if args.max_per_source:
            slices = slices[:args.max_per_source]

        blocked_total = sum(b - a for a, b in blocked)
        free_total = sum(b - a for a, b in free)
        print(f"[{idx}/{len(sources)}] {src.name}")
        print(f"    时长 {total / 60:.1f} 分钟｜扫描 {n_scan} 帧"
              f"｜含人脸 {blocked_total:.0f}s（{blocked_total / total * 100:.1f}%）"
              f"｜可切 {free_total:.0f}s → {len(slices)} 条"
              f"｜耗时 {time.time() - t0:.0f}s")

        src_stats.append({
            "name": src.name, "duration_sec": round(total, 1),
            "face_ratio": round(blocked_total / total, 3),
            "free_sec": round(free_total, 1), "clips": len(slices),
        })
        for start, dur in slices:
            all_plan.append({
                "source": str(src), "source_name": src.name,
                "start_sec": round(start, 3), "duration_sec": round(dur, 3),
            })

    print(f"\n扫描完成，耗时 {(time.time() - t_start) / 60:.1f} 分钟；候选 {len(all_plan)} 条")

    if args.probe:
        return 0

    CANDIDATES_OUT.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "scan": f"{SCAN_W}x{SCAN_H}@{args.fps}fps",
         "source_stats": src_stats, "candidates": all_plan},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选池 → {CANDIDATES_OUT}")

    selected = take_round_robin(all_plan, args.target)
    print(f"首选 {len(selected)} 条（多源轮询）")

    FILM_DIR.mkdir(parents=True, exist_ok=True)
    for old in FILM_DIR.glob("film_*.mp4"):
        old.unlink(missing_ok=True)

    used: set[tuple[str, float]] = set()
    accepted: list[dict] = []
    rejected = 0
    t_cut = time.time()

    def attempt(item: dict) -> bool:
        nonlocal rejected
        dst = FILM_DIR / f"film_{len(accepted):04d}.mp4"
        if not cut_clip(Path(item["source"]), item["start_sec"],
                        item["duration_sec"], dst):
            return False
        if args.audit:
            bad = audit_clip(dst)
            if bad > 0:
                dst.unlink(missing_ok=True)
                rejected += 1
                return False
        item["file"] = dst.name
        item["size_bytes"] = dst.stat().st_size
        accepted.append(item)
        used.add((item["source"], item["start_sec"]))
        return True

    for item in selected:
        if args.target and len(accepted) >= args.target:
            break
        attempt(item)
        if len(accepted) % 50 == 0 and len(accepted):
            print(f"  已接受 {len(accepted)} 条（复核剔除 {rejected}）")

    if args.target and len(accepted) < args.target:
        print(f"首选不足，从候选池补片（当前 {len(accepted)}/{args.target}）...")
        for item in all_plan:
            if len(accepted) >= args.target:
                break
            if (item["source"], item["start_sec"]) in used:
                continue
            attempt(item)

    print(f"\n最终合格 {len(accepted)} 条（复核剔除 {rejected} 条），"
          f"切片耗时 {(time.time() - t_cut) / 60:.1f} 分钟")

    if accepted:
        MAP_OUT.write_text(json.dumps(
            {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "window_sec": args.window, "scan": f"{SCAN_W}x{SCAN_H}@{args.fps}fps",
             "audited": args.audit, "rejected": rejected,
             "source_stats": src_stats, "clips": accepted},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"映射 → {MAP_OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
