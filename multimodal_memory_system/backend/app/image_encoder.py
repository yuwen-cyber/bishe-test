from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)
_model = None

_IMAGE_DIM = 512
_SIZE = 224
# CLIP ViT-B/32 归一化常量
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _load_model():
    global _model
    if _model is None:
        from transformers import CLIPModel

        model_name = "openai/clip-vit-base-patch32"
        # 优先离线加载本地缓存，失败再在线下载（与 embeddings.py 一致）
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            _model = CLIPModel.from_pretrained(model_name)
        except Exception as e:
            logger.warning("Offline load failed (%s), falling back to online download", e)
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _model = CLIPModel.from_pretrained(model_name)
        _model.eval()
    return _model


def _probe_duration(src_path: str) -> float:
    """用 ffprobe 读取视频时长（秒），失败返回 0。"""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", src_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _extract_keyframe_rgb(video_bytes: bytes) -> str:
    """用 ffmpeg 取视频中间一帧，中心裁剪缩放到 224x224，输出 rawvideo rgb24（免 PIL）。"""
    fd, src_path = tempfile.mkstemp(suffix=".webm")
    raw_path = src_path[:-5] + ".rgb"
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
        half = _probe_duration(src_path) / 2.0
        filters = (
            f"scale={_SIZE}:{_SIZE}:force_original_aspect_ratio=increase,"
            f"crop={_SIZE}:{_SIZE}"
        )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{half:.3f}",
                "-i", src_path,
                "-frames:v", "1",
                "-vf", filters,
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                raw_path,
            ],
            check=True,
            capture_output=True,
        )
        return raw_path
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)


def _mock_embed(data: bytes, dim: int) -> list[float]:
    """无模型时的确定性降级向量化：基于字节 md5 哈希，归一化到单位向量。"""
    vec = np.zeros(dim, dtype=np.float32)
    for gram_size in (1, 2, 4):
        for i in range(len(data) - gram_size + 1):
            h = int(hashlib.md5(data[i : i + gram_size]).hexdigest(), 16)
            vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        h = int(hashlib.md5(data or b"").hexdigest(), 16)
        vec[h % dim] = 1.0
    else:
        vec = vec / norm
    return vec.tolist()


def encode_video_keyframe(video_bytes: bytes) -> list[float]:
    """把视频关键帧编码为 512 维 CLIP 图像向量。"""
    if settings.use_real_image:
        try:
            return _encode_real(video_bytes)
        except Exception as e:  # pragma: no cover
            logger.warning("Real image encoding failed, fallback to mock: %s", e)
    return _mock_embed(video_bytes, _IMAGE_DIM)


def _encode_real(video_bytes: bytes) -> list[float]:
    import torch

    _load_model()
    raw_path = _extract_keyframe_rgb(video_bytes)
    try:
        raw = np.fromfile(raw_path, dtype=np.uint8)
        img = raw.reshape(_SIZE, _SIZE, 3).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)  # (3, 224, 224)
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            features = _model.get_image_features(pixel_values=tensor)
        # transformers 5.x 的 get_image_features 返回 BaseModelOutputWithPooling，
        # 图像特征在 pooler_output（已过 visual_projection，512 维）；旧版直接返回张量。
        if hasattr(features, "pooler_output"):
            vec = features.pooler_output
        elif hasattr(features, "image_embeds"):
            vec = features.image_embeds
        else:
            vec = features
        vec = vec.squeeze(0)  # (512,)
        vec = vec / max(float(torch.norm(vec)), 1e-12)
        return vec.tolist()
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


def warmup() -> None:
    if settings.use_real_image:
        try:
            _load_model()
            logger.info("Image encoder model loaded: openai/clip-vit-base-patch32")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load CLIP, fallback to mock: %s", e)
    else:
        logger.info("Image encoder uses mock encoder (dim=%s)", _IMAGE_DIM)