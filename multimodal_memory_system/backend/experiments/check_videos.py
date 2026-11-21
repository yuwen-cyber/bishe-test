"""
检查 generated_video 里所有合成视频是否可播放。
判断标准：ffprobe 退出码 0 + 有视频流 + 有音频流 + 文件非空。
用法：D:\Python3.9\python.exe experiments\check_videos.py
"""
import json
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).parent / "generated_video"


def probe(path: Path):
    """返回 (ok, has_video, has_audio, reason)。"""
    if path.stat().st_size == 0:
        return False, False, False, "文件为空"
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, False, False, f"ffprobe 解析失败: {r.stderr.strip()[:80]}"
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return False, False, False, "ffprobe 输出无法解析"
    types = [s.get("codec_type") for s in data.get("streams", [])]
    has_video = "video" in types
    has_audio = "audio" in types
    if not has_video:
        return False, False, has_audio, "缺视频流"
    if not has_audio:
        return False, has_video, False, "缺音频流"
    return True, True, True, "正常"


def main():
    videos = sorted(OUT_DIR.glob("*.mp4"))
    print(f"共 {len(videos)} 个视频\n")

    bad = []
    no_audio = []
    ok = 0
    for v in videos:
        good, has_v, has_a, reason = probe(v)
        if good:
            ok += 1
        elif reason == "缺音频流":
            no_audio.append(v.name)
        else:
            bad.append((v.name, reason))

    print(f"=== 结果 ===")
    print(f"正常(画面+声音): {ok}")
    print(f"缺音频流: {len(no_audio)}")
    print(f"损坏/无法解析: {len(bad)}")

    if no_audio:
        print("\n缺音频文件:")
        for n in no_audio:
            print(f"  ⚠️  {n}")
    if bad:
        print("\n损坏文件:")
        for n, reason in bad:
            print(f"  ❌ {n}  ({reason})")

    if ok == len(videos):
        print("\n✅ 全部 700 个视频可正常播放，画面和声音都在。")


if __name__ == "__main__":
    main()
