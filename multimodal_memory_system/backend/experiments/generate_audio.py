"""
把 segments_dataset.json 里的文本批量转成情绪语音（Edge TTS，免费）。
用法：
    cd D:\biyesheji\multimodal_memory_system\backend
    python experiments\generate_audio.py

输出：experiments/generated_audio/seg_001.mp3 ... seg_100.mp3
"""
import asyncio
import json
from pathlib import Path

import edge_tts

BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "segments_dataset.json"
OUT_DIR = BASE_DIR / "generated_audio"

# 情绪 → Edge TTS 声音 + 音高(pitch) + 语速(rate)
# 新版 edge-tts 不支持 style 参数，改用 pitch 和 rate 模拟情绪
VOICE_MAP = {
    "喜悦": ("zh-CN-XiaoxiaoNeural", "+20Hz", "+10%"),
    "愤怒": ("zh-CN-YunxiNeural", "+50Hz", "+20%"),
    "悲伤": ("zh-CN-XiaoxiaoNeural", "-20Hz", "-20%"),
    "恐惧": ("zh-CN-YunjianNeural", "+10Hz", "+25%"),
    "惊讶": ("zh-CN-XiaoyiNeural", "+30Hz", "+10%"),
    "厌恶": ("zh-CN-YunxiNeural", "-10Hz", "-15%"),
    "中性": ("zh-CN-YunyangNeural", "+0Hz", "+0%"),
}


async def gen_one(text: str, emotion: str, out_path: Path):
    voice, pitch, rate = VOICE_MAP.get(emotion, ("zh-CN-XiaoxiaoNeural", "+0Hz", "+0%"))
    tts = edge_tts.Communicate(
        text=text,
        voice=voice,
        pitch=pitch,
        rate=rate,
    )
    await tts.save(str(out_path))


async def main():
    if not DATASET_FILE.exists():
        print(f"❌ 找不到 {DATASET_FILE}，先跑 clean_dataset.py")
        return

    segments = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    total = len(segments)
    ok = 0
    fail = 0
    for i, seg in enumerate(segments, 1):
        sid = seg["segment_id"]
        text = seg["asr_text"]
        emotion = seg.get("emotion", "中性")
        out_path = OUT_DIR / f"seg_{sid:03d}.mp3"

        if out_path.exists():
            print(f"[{i}/{total}] seg_{sid:03d} 已存在，跳过")
            ok += 1
            continue

        try:
            await gen_one(text, emotion, out_path)
            print(f"[{i}/{total}] seg_{sid:03d} ({emotion}) -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{total}] seg_{sid:03d} 失败: {e}")
            fail += 1

    print(f"\n✅ 完成: 成功 {ok} 条, 失败 {fail} 条")
    print(f"   输出目录: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
