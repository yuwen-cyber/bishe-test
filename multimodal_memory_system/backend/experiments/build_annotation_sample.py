#!/usr/bin/env python3
"""实验 1（可做部分）：构建情绪标注任务清单（盲标抽样）。

导师要求：情绪标注由 2~3 人独立标注并报告 Kappa。
本脚本只负责**抽样与出题**，不做任何标注、不生成任何评分。

关键设计：
- **分层抽样**：按 7 类情绪在库中的比例抽（每类至少 2 条），保证稀有类（厌恶 27、中性 42）也有样本；
- **盲标**：任务清单**不含原始 emotion 列**，标注者看不到模型/原标结果，避免锚定；
- **答案另存**：原始标签单独写入 answer_key.csv，供后续算「标注者一致性」和「模型-人一致率」，
  该文件不给标注者看；
- **可复算**：固定随机种子，重跑得到同一批样本。

输出（默认 experiments/annotation/）：
  tasks.csv        待标注清单（含 presigned 音视频链接、ASR 文本，无原始情绪列）
  answer_key.csv   原始情绪标签（分析用，标注者勿看）
  sampling_log.md  抽样说明（各类数量、种子、生成时间）

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\build_annotation_sample.py --size 60 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app import storage  # noqa: E402
from app.database import SessionLocal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建情绪标注抽样任务")
    parser.add_argument("--size", type=int, default=60, help="抽样总条数（建议 >=50，Kappa 才稳）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/annotation")
    parser.add_argument("--expires-minutes", type=int, default=240, help="音视频链接有效期")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    rows = db.execute(
        text(
            "SELECT s.id AS segment_id, s.conversation_id, s.emotion, s.asr_text, "
            "c.audio_object_key, c.video_object_key, c.duration_ms "
            "FROM segments s JOIN conversations c ON s.conversation_id = c.id "
            "ORDER BY s.id"
        )
    ).mappings().all()
    db.close()

    if not rows:
        print("[错误] 库里没有片段")
        return 1

    rng = random.Random(args.seed)
    by_emotion: dict[str, list] = defaultdict(list)
    for r in rows:
        by_emotion[str(r["emotion"] or "未标注")].append(r)

    total = len(rows)
    # 按比例分配，每类至少 2 条，且不超过该类总数
    alloc: dict[str, int] = {}
    for emo, items in by_emotion.items():
        n = max(2, round(args.size * len(items) / total))
        alloc[emo] = min(n, len(items))
    # 数量对齐到 --size（多则从最大类扣，少则补给最大类）
    diff = args.size - sum(alloc.values())
    biggest = max(alloc, key=lambda e: alloc[e])
    alloc[biggest] = max(2, min(len(by_emotion[biggest]), alloc[biggest] + diff))

    sample = []
    for emo, n in alloc.items():
        sample.extend(sorted(rng.sample(by_emotion[emo], n), key=lambda x: int(x["segment_id"])))

    tasks = []
    keys = []
    for idx, r in enumerate(sample, start=1):
        audio_url = video_url = ""
        try:
            audio_url = storage.presigned_url(r["audio_object_key"], expires_minutes=args.expires_minutes) \
                if r["audio_object_key"] else ""
        except Exception:
            pass
        try:
            video_url = storage.presigned_url(r["video_object_key"], expires_minutes=args.expires_minutes) \
                if r["video_object_key"] else ""
        except Exception:
            pass

        tasks.append({
            "task_id": idx,
            "segment_id": int(r["segment_id"]),
            "conversation_id": int(r["conversation_id"]),
            "asr_text": (r["asr_text"] or "").strip(),
            "duration_ms": int(r["duration_ms"] or 0),
            "audio_url": audio_url,
            "video_url": video_url,
        })
        keys.append({
            "segment_id": int(r["segment_id"]),
            "original_emotion": str(r["emotion"] or ""),
        })

    with (out_dir / "tasks.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
        writer.writeheader()
        writer.writerows(tasks)
    with (out_dir / "answer_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys[0].keys()))
        writer.writeheader()
        writer.writerows(keys)

    md = [
        "# 情绪标注抽样说明\n",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 随机种子：{args.seed}（重跑可复现同一批样本）",
        f"- 总条数：{len(sample)}（全库 {total} 条）",
        f"- 音视频链接有效期：{args.expires_minutes} 分钟（过期后用 annotate_cli.py 重新生成）\n",
        "## 各情绪抽样数\n",
        "| 情绪 | 库内条数 | 抽样条数 |",
        "|---|---|---|",
    ]
    for emo in sorted(alloc, key=lambda e: -len(by_emotion[e])):
        md.append(f"| {emo} | {len(by_emotion[emo])} | {alloc[emo]} |")
    md.append("")
    md.append("## 盲标约定\n")
    md.append("- `tasks.csv` **不含**原始情绪列，标注者只看 ASR 文本 + 听音频（必要时看视频）。")
    md.append("- `answer_key.csv` 为原始标签，仅用于事后分析，**标注过程中不得打开**。")
    (out_dir / "sampling_log.md").write_text("\n".join(md), encoding="utf-8")

    print(f"已生成 {len(tasks)} 条任务 -> {out_dir / 'tasks.csv'}")
    for emo in sorted(alloc, key=lambda e: -len(by_emotion[e])):
        print(f"  {emo}: 库内 {len(by_emotion[emo])}，抽 {alloc[emo]}")
    print(f"答案键 -> {out_dir / 'answer_key.csv'}（标注者勿看）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
