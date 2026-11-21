#!/usr/bin/env python3
"""灌入假片段数据：走 /api/upload，用 ffmpeg 现场合成 1 秒 webm 视频/音频。

用法：
    python experiments/seed_demo_data.py --base-url http://localhost:8000 --count 20

说明：
- 不 import app 内部模块，全部走 HTTP。
- 视频：1 秒纯黑 VP8/webm；音频：1 秒静音 Opus/webm（由本机 ffmpeg 合成）。
- mock ASR 按 `len(audio_bytes) % 5` 从 5 句固定文案中选取一句；本脚本会按顺序给
  音频补齐若干字节，让 20 条数据依次命中 5 句不同文案（每句 4 条），获得可区分的文本。
- 推荐的 .env 组合：USE_REAL_ASR=false、USE_REAL_EMOTION=false、USE_REAL_EMBEDDING=true
  （embedding 用真实 bge，这样向量检索才有真实语义区分度；ASR/情绪用 mock 保证伪造音频可跑通）。
- 上传成功后会把 (segment_id, asr_text, emotion, conversation_id) 写入
  experiments/output/seed_mapping.json，方便回填 query_set.json 的 relevant_segment_ids。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests


def _make_webm(out_path: Path, kind: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法合成测试媒体文件；请先安装 ffmpeg 或手动提供视频/音频文件")

    if kind == "audio":
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "1",
            "-c:a", "libopus", "-b:a", "16k",
            str(out_path),
        ]
    else:  # video
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25",
            "-t", "1",
            "-c:v", "libvpx", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="灌入假片段数据")
    parser.add_argument("--base-url", default="http://localhost:8000", help="后端 base URL")
    parser.add_argument("--count", type=int, default=20, help="灌入条数")
    parser.add_argument("--output-dir", default="experiments/output", help="输出目录")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = tmp_dir / "demo_audio.webm"
        video_path = tmp_dir / "demo_video.webm"
        _make_webm(audio_path, "audio")
        _make_webm(video_path, "video")

        base_audio = audio_path.read_bytes()
        base_video = video_path.read_bytes()

        mapping: list[dict] = []
        for i in range(1, args.count + 1):
            # 让音频字节长度对 5 取模 = (i-1)%5，从而命中 5 句不同 mock 文案
            residue = (i - 1) % 5
            pad = (residue - len(base_audio) % 5) % 5
            audio_bytes = base_audio + b"\x00" * pad
            try:
                resp = requests.post(
                    f"{base_url}/api/upload",
                    files={
                        "video": ("demo.webm", base_video, "video/webm"),
                        "audio": ("demo.webm", audio_bytes, "audio/webm"),
                    },
                    data={"duration_ms": "1000", "is_demo": "true"},
                    timeout=180,
                )
                resp.raise_for_status()
                body = resp.json()
                mapping.append({
                    "upload_index": i,
                    "segment_id": body.get("segment_id"),
                    "conversation_id": body.get("conversation_id"),
                    "emotion": body.get("emotion"),
                    "asr_text": body.get("asr_text"),
                })
                print(
                    f"[{i:>2}/{args.count}] segment_id={body.get('segment_id')} "
                    f"emotion={body.get('emotion')} text={body.get('asr_text')!r}"
                )
            except Exception as exc:
                print(f"[{i:>2}/{args.count}] FAILED: {exc}", file=sys.stderr)

    mapping_path = output_dir / "seed_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    seg_ids = [m["segment_id"] for m in mapping if m.get("segment_id") is not None]
    print(f"\n完成：成功 {len(seg_ids)} 条。segment_id 列表：{seg_ids}")
    print(f"映射已写入：{mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())