"""
测试：把"喜悦"类的 TTS 音频和 happy 人脸视频合并。
用法：python merge_test.py
"""
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(r"D:\biyesheji\multimodal_memory_system\backend\experiments")
DATASET = BASE_DIR / "segments_dataset.json"
AUDIO_DIR = BASE_DIR / "generated_audio"
VIDEO_DIR = Path(r"D:\biyesheji\face_videos")
OUT_DIR = BASE_DIR / "generated_video"

# 情绪 → 视频文件夹前缀（现在只有 happy）
EMOTION_VIDEO_MAP = {
    "喜悦": "happy",
}

def get_video_for_emotion(emotion: str, idx: int) -> Path:
    """按情绪挑一个视频，循环复用。"""
    prefix = EMOTION_VIDEO_MAP.get(emotion)
    if not prefix:
        return None
    videos = sorted(VIDEO_DIR.glob(f"{prefix}*.mp4"))
    if not videos:
        return None
    return videos[idx % len(videos)]

def merge_one(video_path: Path, audio_path: Path, out_path: Path):
    """用 ffmpeg 合并视频和音频：强制用视频画面 + mp3 声音。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",      # 强制用视频文件的视频流
        "-map", "1:a:0",      # 强制用 mp3 的音频流
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def main():
    segments = json.loads(DATASET.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    # 只处理喜悦类
    joy_segments = [s for s in segments if s.get("emotion") == "喜悦"]
    print(f"找到 {len(joy_segments)} 条喜悦片段")

    ok = 0
    for i, seg in enumerate(joy_segments):
        sid = seg["segment_id"]
        audio = AUDIO_DIR / f"seg_{sid:03d}.mp3"
        if not audio.exists():
            print(f"  跳过 seg_{sid:03d}，音频不存在")
            continue

        video = get_video_for_emotion("喜悦", i)
        if not video:
            print(f"  跳过 seg_{sid:03d}，没找到 happy 视频")
            continue

        out = OUT_DIR / f"video_{sid:03d}.mp4"
        try:
            merge_one(video, audio, out)
            print(f"  ✅ seg_{sid:03d} + {video.name} -> {out.name}")
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  ❌ seg_{sid:03d} 失败: {e}")

    print(f"\n完成: {ok} 条喜悦类视频已合并到 {OUT_DIR}")
    print("去这个文件夹点开几个视频看看效果对不对")

if __name__ == "__main__":
    main()
