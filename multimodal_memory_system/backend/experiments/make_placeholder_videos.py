"""
把 placeholder.mp4 复制成 100 个占位视频，和 generated_audio/ 里的 mp3 一一对应。
用法：python make_placeholder_videos.py
"""
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / "segments_dataset.json"
PLACEHOLDER = BASE_DIR / "placeholder.mp4"
OUT_DIR = BASE_DIR / "generated_video"

def main():
    if not PLACEHOLDER.exists():
        print("❌ 先跑上面那条 ffmpeg 命令生成 placeholder.mp4")
        return

    segments = json.loads(DATASET.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    for seg in segments:
        sid = seg["segment_id"]
        out = OUT_DIR / f"video_{sid:03d}.mp4"
        shutil.copy(PLACEHOLDER, out)

    print(f"✅ 已生成 {len(segments)} 个占位视频到 {OUT_DIR}")

if __name__ == "__main__":
    main()
