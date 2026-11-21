#!/usr/bin/env python3
"""实验 1：命令行情绪标注工具（盲标、可断点续标）。

用法（backend 目录，任选一名标注者名）：
    .venv\\Scripts\\python.exe experiments\\annotate_cli.py --annotator 张三
    .venv\\Scripts\\python.exe experiments\\annotate_cli.py --annotator 李四

交互命令：
    1~7            选择情绪标签（1 喜悦 / 2 惊讶 / 3 愤怒 / 4 恐惧 / 5 悲伤 / 6 中性 / 7 厌恶）
    p              用默认浏览器播放该条音频
    v              播放该条视频（含口型与表情，可辅助判断）
    s              跳过本条（不作为有效标注）
    q              保存并退出（下次接着标）
    h              查看说明

标注结果实时追加到 experiments/annotation/annotations_<标注者>.jsonl，
退出时另存一份 CSV（供 compute_kappa.py 使用）。中断不丢进度。

设计要点：
- 工具**不显示**原始情绪标签，也不显示他人标注结果（盲标、独立标注）；
- 每条记录写入标注者名与耗时，便于事后核查"是否独立完成"；
- 只允许 7 个固定标签，不接受自由文本，避免标签集合漂移。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

LABELS = {
    "1": "喜悦",
    "2": "惊讶",
    "3": "愤怒",
    "4": "恐惧",
    "5": "悲伤",
    "6": "中性",
    "7": "厌恶",
}

HELP = """
可用命令：
  1 喜悦   2 惊讶   3 愤怒   4 恐惧   5 悲伤   6 中性   7 厌恶
  p 播放音频   v 播放视频   s 跳过   q 保存退出   h 帮助

判读约定（摘要，详见 emotion_labeling_protocol.md）：
  - 以说话人**实际表达的情绪**为准，不以文本字面褒贬机械判定；
  - 同时出现多种情绪时，取**主导情绪**（语气最强烈的那一种）；
  - 完全平淡、听不出情绪倾向 → 中性；
  - 视频与音频矛盾时，以音频韵律为主、文本内容为辅。
"""


def _load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_done(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label"):
                done.add(int(rec["task_id"]))
    return done


def _open_url(url: str) -> bool:
    if not url:
        print("  [提示] 该条没有可用的媒体链接（可能对象缺失）")
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # noqa: S606 - 仅打开本地预设的 presigned URL
        elif sys.platform == "darwin":
            os.system(f'open "{url}"')  # noqa: S605
        else:
            os.system(f'xdg-open "{url}"')  # noqa: S605
        return True
    except Exception as exc:
        print(f"  [提示] 打开失败：{exc}")
        return False


def _export_csv(jsonl_path: Path, csv_path: Path, annotator: str) -> int:
    rows: list[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows = [r for r in rows if r.get("label")]
    if not rows:
        return 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["annotator", "task_id", "segment_id", "label", "seconds_spent"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "annotator": r.get("annotator", annotator),
                "task_id": r["task_id"],
                "segment_id": r["segment_id"],
                "label": r["label"],
                "seconds_spent": round(float(r.get("seconds_spent", 0.0)), 1),
            })
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="命令行情绪标注工具")
    parser.add_argument("--annotator", required=True, help="标注者代号（写入结果，便于区分）")
    parser.add_argument("--tasks", default="experiments/annotation/tasks.csv")
    parser.add_argument("--out-dir", default="experiments/annotation")
    parser.add_argument("--open-audio", action="store_true", help="自动播放音频（默认手动按 p）")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"[错误] 找不到任务清单 {tasks_path}，先运行 build_annotation_sample.py")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"annotations_{args.annotator}.jsonl"
    csv_path = out_dir / f"annotations_{args.annotator}.csv"

    tasks = _load_tasks(tasks_path)
    done = _load_done(jsonl_path)

    print(f"标注者：{args.annotator}")
    print(f"任务总数 {len(tasks)}，已完成 {len(done)}，本次待标 {len(tasks) - len(done)}")
    print(HELP)

    labeled_now = 0
    for task in tasks:
        tid = int(task["task_id"])
        if tid in done:
            continue

        print("=" * 72)
        print(f"[{tid}/{len(tasks)}] segment_id={task['segment_id']}  "
              f"时长 {int(task['duration_ms']) / 1000:.1f}s")
        print(f"文本：{task['asr_text']}")
        if args.open_audio:
            _open_url(task["audio_url"])

        t0 = time.perf_counter()
        while True:
            cmd = input("命令(1-7/p/v/s/q/h) > ").strip().lower()
            if cmd in LABELS:
                rec = {
                    "annotator": args.annotator,
                    "task_id": tid,
                    "segment_id": int(task["segment_id"]),
                    "label": LABELS[cmd],
                    "seconds_spent": time.perf_counter() - t0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                labeled_now += 1
                print(f"  -> 记为「{LABELS[cmd]}」")
                break
            if cmd == "p":
                _open_url(task["audio_url"])
            elif cmd == "v":
                _open_url(task["video_url"])
            elif cmd == "s":
                print("  -> 已跳过（不计入有效标注）")
                break
            elif cmd == "q":
                n = _export_csv(jsonl_path, csv_path, args.annotator)
                print(f"\n已保存 {n} 条 -> {csv_path}")
                return 0
            elif cmd == "h":
                print(HELP)
            else:
                print("  无法识别的命令，输入 h 查看说明")

    n = _export_csv(jsonl_path, csv_path, args.annotator)
    print(f"\n本次新增 {labeled_now} 条，累计有效标注 {n} 条 -> {csv_path}")
    print("下一步：让第二名标注者用同样的命令跑一遍（不同 --annotator），"
          "然后用 compute_kappa.py 计算一致性。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
