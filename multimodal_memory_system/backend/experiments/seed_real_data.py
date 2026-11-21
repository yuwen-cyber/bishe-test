#!/usr/bin/env python3
"""把 experiments 下的 700 条真实音视频灌入后端系统（走 HTTP /api/upload）。

【做了什么】
- 读取 segments_dataset.json（segment_id, scenario, emotion, asr_text, duration_ms）；
- 逐条把 generated_audio/seg_xxx.mp3 + generated_video/video_xxx.mp4 通过
  POST /api/upload 上传；后端内部自行跑 ASR -> 情绪识别 -> 文本向量化 -> 写 MySQL/Milvus；
- 记录“原始编号 -> 数据库 segment_id / conversation_id”映射，落盘到
  experiments/output/seed_mapping_real.json，供回填 query_set.json 使用。

【重要说明】
- 入库的 asr_text / emotion 是后端模型重新识别的结果，可能与数据集标注真值有偏差，
  这是现有上传接口的行为，本脚本不改后端代码。
- 音频/图像向量当前按 .env 开关走 mock 降级，不影响灌库流程。
- 支持断点续传：已在映射表中的原始编号会自动跳过；中断后重跑即可接着灌。

用法（在 backend 目录下，使用后端 .venv 解释器）：
    .venv\\Scripts\\python.exe experiments\\seed_real_data.py
可选参数：
    --base-url http://localhost:8000
    --limit 5          # 只灌前 N 条（试跑用）
    --retry 1          # 单条失败后的重试次数
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENTS_DIR / "segments_dataset.json"
AUDIO_DIR = EXPERIMENTS_DIR / "generated_audio"
VIDEO_DIR = EXPERIMENTS_DIR / "generated_video"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
MAPPING_PATH = OUTPUT_DIR / "seed_mapping_real.json"
FAILED_PATH = OUTPUT_DIR / "seed_failed.json"


def _load_mapping() -> list[dict]:
    if MAPPING_PATH.exists():
        try:
            return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_mapping(mapping: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="灌入 700 条真实音视频数据")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=0, help="只灌前 N 条，0 表示全部")
    parser.add_argument("--retry", type=int, default=1, help="单条失败重试次数")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    mapping = _load_mapping()
    done_orig = {int(m["orig_segment_id"]) for m in mapping}

    # 预检：数据集里编号 -> 文件是否齐全
    total = len(dataset)
    plan: list[dict] = []
    missing_files: list[dict] = []
    for item in dataset:
        orig_id = int(item["segment_id"])
        audio_path = AUDIO_DIR / f"seg_{orig_id:03d}.mp3"
        video_path = VIDEO_DIR / f"video_{orig_id:03d}.mp4"
        if not audio_path.exists() or not video_path.exists():
            missing_files.append({"orig_segment_id": orig_id,
                                  "audio": str(audio_path.exists()),
                                  "video": str(video_path.exists())})
            continue
        plan.append({"item": item, "audio": audio_path, "video": video_path})

    if missing_files:
        print(f"[警告] {len(missing_files)} 条缺音/视频文件，已跳过：")
        for m in missing_files:
            print("   ", m)

    if args.limit and args.limit > 0:
        plan = plan[:args.limit]

    todo = [p for p in plan if int(p["item"]["segment_id"]) not in done_orig]
    print(f"数据集共 {total} 条；已灌入 {len(done_orig)} 条；本次待灌 {len(todo)} 条。")

    failed: list[dict] = []
    session = requests.Session()
    t0 = time.time()

    for idx, p in enumerate(todo, start=1):
        item = p["item"]
        orig_id = int(item["segment_id"])
        audio_path: Path = p["audio"]
        video_path: Path = p["video"]
        duration_ms = int(item.get("duration_ms") or 0)
        labeled_emotion = item.get("emotion")
        scenario = item.get("scenario")

        last_err = ""
        body: dict | None = None
        for attempt in range(args.retry + 1):
            try:
                audio_bytes = audio_path.read_bytes()
                video_bytes = video_path.read_bytes()
                resp = session.post(
                    f"{base_url}/api/upload",
                    files={
                        "video": (video_path.name, video_bytes, "video/mp4"),
                        "audio": (audio_path.name, audio_bytes, "audio/mpeg"),
                    },
                    data={
                        "duration_ms": str(duration_ms),
                        "is_demo": "false",
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                body = resp.json()
                break
            except Exception as exc:  # 单条失败不中断整体
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < args.retry:
                    time.sleep(2)

        if body is None:
            failed.append({
                "orig_segment_id": orig_id,
                "labeled_emotion": labeled_emotion,
                "scenario": scenario,
                "error": last_err,
            })
            print(f"[{idx:>3}/{len(todo)}] orig={orig_id} FAILED: {last_err}", file=sys.stderr)
            continue

        record = {
            "orig_segment_id": orig_id,
            "db_segment_id": body.get("segment_id"),
            "conversation_id": body.get("conversation_id"),
            "labeled_emotion": labeled_emotion,
            "backend_emotion": body.get("emotion"),
            "scenario": scenario,
            "duration_ms": duration_ms,
            "backend_asr_text": body.get("asr_text"),
            "audio_file": audio_path.name,
            "video_file": video_path.name,
        }
        mapping.append(record)
        _save_mapping(mapping)  # 每条落盘，中断不丢进度

        elapsed = time.time() - t0
        rate = idx / elapsed if elapsed > 0 else 0
        print(
            f"[{idx:>3}/{len(todo)}] orig={orig_id} db_seg={body.get('segment_id')} "
            f"conv={body.get('conversation_id')} 标注={labeled_emotion} "
            f"识别={body.get('emotion')} ({rate:.1f} 条/分)"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_PATH.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n========== 灌库结束 ==========")
    print(f"本次成功 {len(todo) - len(failed)} 条，失败 {len(failed)} 条")
    print(f"累计映射记录 {len(mapping)} 条 -> {MAPPING_PATH}")
    if failed:
        print(f"失败清单 -> {FAILED_PATH}（修复后重跑本脚本即可续传）")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
