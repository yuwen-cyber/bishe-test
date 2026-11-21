"""
全情绪视频合并脚本：把 TTS 音频和对应情绪的人脸视频合并。
用法：python merge_all.py
支持断点续传，已经生成的视频会跳过。
"""
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(r"D:\biyesheji\multimodal_memory_system\backend\experiments")
DATASET = BASE_DIR / "segments_dataset.json"
AUDIO_DIR = BASE_DIR / "generated_audio"
VIDEO_DIR = Path(r"D:\biyesheji\face_videos")
OUT_DIR = BASE_DIR / "generated_video"

# 情绪 → 视频文件名前缀
EMOTION_VIDEO_MAP = {
    "喜悦": "happy",
    "悲伤": "sad",
    "愤怒": "angry",
    "恐惧": "scared",
    "惊讶": "surprised",
    "厌恶": "disgusted",
    "中性": "expressionless",
}


def get_video_pool(emotion: str) -> list[Path]:
    """返回该情绪下所有可用视频。"""
    prefix = EMOTION_VIDEO_MAP.get(emotion)
    if not prefix:
        return []
    return sorted(VIDEO_DIR.glob(f"{prefix}*.mp4"))


def merge_one(video_path: Path, audio_path: Path, out_path: Path):
    """用 ffmpeg 合并：强制用视频画面 + mp3 声音。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
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

    # 按情绪分组统计
    emotion_count: dict[str, int] = {}
    for s in segments:
        emo = s.get("emotion", "未知")
        emotion_count[emo] = emotion_count.get(emo, 0) + 1

    print("数据集情绪分布:")
    for emo, cnt in emotion_count.items():
        pool = get_video_pool(emo)
        status = f"视频池 {len(pool)} 个" if pool else "⚠️ 还没下视频"
        print(f"  {emo}: {cnt} 条  ({status})")

    ok = 0
    skipped = 0
    no_video = 0
    no_audio = 0
    fail = 0

    for seg in segments:
        sid = seg["segment_id"]
        emotion = seg.get("emotion", "中性")
        out_path = OUT_DIR / f"video_{sid:03d}.mp4"

        # 已经生成过就跳过
        if out_path.exists():
            skipped += 1
            continue

        audio_path = AUDIO_DIR / f"seg_{sid:03d}.mp3"
        if not audio_path.exists():
            print(f"  ⚠️  seg_{sid:03d} 没有音频文件")
            no_audio += 1
            continue

        pool = get_video_pool(emotion)
        if not pool:
            no_video += 1
            continue

        video_path = pool[sid % len(pool)]
        try:
            merge_one(video_path, audio_path, out_path)
            print(f"  ✅ seg_{sid:03d} ({emotion}) + {video_path.name}")
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  ❌ seg_{sid:03d} 失败")
            fail += 1

    print(f"\n=== 汇总 ===")
    print(f"  新生成: {ok}")
    print(f"  已存在跳过: {skipped}")
    print(f"  缺音频: {no_audio}")
    print(f"  缺视频素材: {no_video}")
    print(f"  合并失败: {fail}")
    print(f"  输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
