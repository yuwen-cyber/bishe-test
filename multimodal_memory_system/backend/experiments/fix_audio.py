"""
补生成损坏/空的 TTS 音频（0 字节文件）。
用法：python fix_audio.py
"""
import asyncio
import json
from pathlib import Path

import edge_tts

BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "segments_dataset.json"
OUT_DIR = BASE_DIR / "generated_audio"

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
    tts = edge_tts.Communicate(text=text, voice=voice, pitch=pitch, rate=rate)
    await tts.save(str(out_path))


async def main():
    segments = {s["segment_id"]: s for s in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}

    # 找出所有 0 字节的 mp3
    empty = [p for p in OUT_DIR.glob("*.mp3") if p.stat().st_size == 0]
    print(f"发现 {len(empty)} 个空音频: {[p.name for p in empty]}")

    fixed = 0
    fail = 0
    for p in empty:
        sid = int(p.stem.split("_")[1])
        seg = segments.get(sid)
        if not seg:
            print(f"  ❌ {p.name} 在数据集里找不到")
            fail += 1
            continue
        text = seg["asr_text"]
        emotion = seg.get("emotion", "中性")
        try:
            await gen_one(text, emotion, p)
            size = p.stat().st_size
            if size == 0:
                print(f"  ❌ {p.name} 重新生成后仍是空文件")
                fail += 1
            else:
                print(f"  ✅ {p.name} ({emotion}) 重新生成成功, {size} 字节")
                fixed += 1
        except Exception as e:
            print(f"  ❌ {p.name} 失败: {e}")
            fail += 1

    print(f"\n✅ 补音频完成: 修复 {fixed}, 失败 {fail}")


if __name__ == "__main__":
    asyncio.run(main())
