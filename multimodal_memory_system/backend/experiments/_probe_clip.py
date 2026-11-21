#!/usr/bin/env python3
"""临时探查：transformers 5.x 下 CLIP 图像嵌入的正确取法。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from app import config  # noqa: F401
from app import image_encoder
from transformers import CLIPModel

VIDEO = Path(__file__).resolve().parent / "generated_video" / "video_000.mp4"
_SIZE = 224
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def main() -> int:
    video_bytes = VIDEO.read_bytes()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    raw_path = image_encoder._extract_keyframe_rgb(video_bytes)
    try:
        raw = np.fromfile(raw_path, dtype=np.uint8)
        img = raw.reshape(_SIZE, _SIZE, 3).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)  # (3,224,224)
        tensor = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=tensor)
        print("返回类型:", type(out).__name__)
        feat = out.pooler_output
        print("pooler_output 形状:", tuple(feat.shape))
        vec = feat.squeeze(0)
        vec = vec / max(float(torch.norm(vec)), 1e-12)
        print("归一化后维度:", tuple(vec.shape), "范数:", float(torch.norm(vec)))
    finally:
        import os
        if os.path.exists(raw_path):
            os.remove(raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
