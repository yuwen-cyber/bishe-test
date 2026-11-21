#!/usr/bin/env python3
"""临时验证：Wav2Vec2 音频编码 + CLIP 图像关键帧编码是否可跑通。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\_verify_real_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# 先触发 config，设置 HF_ENDPOINT 镜像
from app import config  # noqa: F401
from app import audio_encoder, image_encoder

AUDIO = Path(__file__).resolve().parent / "generated_audio" / "seg_000.mp3"
VIDEO = Path(__file__).resolve().parent / "generated_video" / "video_000.mp4"


def main() -> int:
    audio_bytes = AUDIO.read_bytes()
    video_bytes = VIDEO.read_bytes()
    print(f"音频字节={len(audio_bytes)}  视频字节={len(video_bytes)}")

    t0 = time.perf_counter()
    audio_vec = audio_encoder._encode_real(audio_bytes)
    t_audio = time.perf_counter() - t0
    av = np.asarray(audio_vec, dtype=float)
    print(f"[音频] 维度={av.shape[0]} L2范数={np.linalg.norm(av):.4f} 耗时={t_audio:.2f}s")

    t0 = time.perf_counter()
    image_vec = image_encoder._encode_real(video_bytes)
    t_image = time.perf_counter() - t0
    iv = np.asarray(image_vec, dtype=float)
    print(f"[图像] 维度={iv.shape[0]} L2范数={np.linalg.norm(iv):.4f} 耗时={t_image:.2f}s")

    if av.shape[0] == 768 and iv.shape[0] == 512 and np.linalg.norm(av) > 0 and np.linalg.norm(iv) > 0:
        print("\n验证通过：两路真模型均可加载并输出归一化向量。")
        return 0
    print("\n验证失败：维度或范数异常。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
