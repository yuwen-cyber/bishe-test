#!/usr/bin/env python3
"""数据总览：一眼看清库里到底有哪些数据（正式 700 / 噪声 1300 / 用户自录）。

答辩时被问"你新加的数据在哪、噪声长什么样"，直接跑这个脚本即可。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\show_dataset.py
    .venv\\Scripts\\python.exe experiments\\show_dataset.py --sample 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
logging.disable(logging.WARNING)

from app import milvus_client, models  # noqa: E402
from app.database import SessionLocal  # noqa: E402

EXPERIMENTS = BACKEND_DIR / "experiments"
NOISE_MAPPING = EXPERIMENTS / "output" / "batch_import_mapping.json"
REAL_VIDEO_DIR = EXPERIMENTS / "real_videos"
NOISE_VIDEO_DIR = EXPERIMENTS / "noise" / "video"
NOISE_AUDIO_DIR = EXPERIMENTS / "noise" / "audio"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=3, help="每类抽样条数")
    args = parser.parse_args()

    db = SessionLocal()
    segs = db.query(models.Segment).order_by(models.Segment.id).all()
    total = len(segs)

    noise_ids = set()
    noise_group: dict[int, str] = {}
    if NOISE_MAPPING.exists():
        for m in json.loads(NOISE_MAPPING.read_text(encoding="utf-8")):
            sid = m.get("db_segment_id")
            if sid:
                noise_ids.add(int(sid))
                noise_group[int(sid)] = m.get("group") or ""

    formal = [s for s in segs if int(s.id) <= 700]
    noise = [s for s in segs if int(s.id) in noise_ids]
    recorded = [s for s in segs if s.is_demo is False and int(s.id) not in noise_ids]

    print("=" * 64)
    print(f"总条数 {total} ｜ 正式 {len(formal)} ｜ 噪声 {len(noise)} ｜ 用户自录 {len(recorded)}")
    print(f"id 范围 {min(int(s.id) for s in segs)} ~ {max(int(s.id) for s in segs)}")
    print("=" * 64)

    print("\n【区分规则】")
    print("  正式 700 条：id 1~700（最早灌入的真实录音转写）")
    print(f"  噪声 1300 条：id {min(noise_ids)}~{max(noise_ids)} 区间内的 {len(noise_ids)} 个 id（TTS 合成对白 + 人脸切片）")
    print("  用户自录：is_demo=False（自己录的音，目前 %d 条）" % len(recorded))

    # 情绪分布
    for name, group in (("正式", formal), ("噪声", noise)):
        c = Counter(s.emotion for s in group)
        print(f"\n【{name}情绪分布】")
        for emo, n in c.most_common():
            print(f"  {emo:4s} {n:4d}  ({n/len(group)*100:.1f}%)")

    # 噪声的 7 类配比
    if noise_group:
        c = Counter(noise_group.values())
        print("\n【噪声配比（按生成时标注）】")
        for g, n in c.most_common():
            print(f"  {g:4s} {n:4d}")

    # 抽样对比
    random.seed(42)
    print(f"\n【随机抽样：正式 {args.sample} 条】")
    for s in random.sample(formal, min(args.sample, len(formal))):
        print(f"  #{s.id} [{s.emotion}] {(s.asr_text or '')[:56]}")
    if noise:
        print(f"\n【随机抽样：噪声 {args.sample} 条】")
        for s in random.sample(noise, min(args.sample, len(noise))):
            print(f"  #{s.id} [{s.emotion}|{noise_group.get(int(s.id),'')}] {(s.asr_text or '')[:56]}")

    # 向量与文件
    print("\n【向量库】")
    try:
        col = milvus_client._get_collection()
        col.load()
        rows = col.query(expr="segment_id >= 0", output_fields=["segment_id"], limit=16384)
        print(f"  Milvus 可读行 {len(rows)}（num_entities {col.num_entities}，差值含已删墓碑）")
    except Exception as e:
        print(f"  Milvus 读取失败：{e}")

    print("\n【音视频文件（MinIO 原始对象之外的本地素材）】")
    for label, path, suffix in (
        ("正式真实切片 real_videos/*.mp4", REAL_VIDEO_DIR, "*.mp4"),
        ("噪声视频 noise/video/*.mp4", NOISE_VIDEO_DIR, "*.mp4"),
        ("噪声音频 noise/audio/*.mp3", NOISE_AUDIO_DIR, "*.mp3"),
    ):
        n = len(list(path.glob(suffix))) if path.exists() else 0
        print(f"  {label}: {n} 个")

    print("\n【前端怎么看到它们】")
    print("  1) /search 检索对比：搜'压力''加班''吃不下饭'等，噪声片段会混在结果里（前端不标注是否为噪声）")
    print("  2) 结果卡片可播放音视频：正式 700 条是真实人脸切片，噪声 1300 条是合成对白配音 + 人脸切片")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
