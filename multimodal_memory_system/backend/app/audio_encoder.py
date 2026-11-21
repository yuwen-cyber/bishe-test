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
_feature_extractor = None

_AUDIO_DIM = 768


def _to_wav_16k(audio_bytes: bytes) -> str:
    """用 ffmpeg 把输入音频转成 16k 单声道 wav，供 Wav2Vec2 编码。"""
    fd, src_path = tempfile.mkstemp(suffix=".webm")
    wav_path = src_path[:-5] + ".wav"
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
        return wav_path
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)


def _load_model():
    global _model, _feature_extractor
    if _model is None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_name = "facebook/wav2vec2-base-960h"
        # 优先离线加载本地缓存，失败再在线下载（与 embeddings.py 一致）
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            _model = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            logger.warning("Offline load failed (%s), falling back to online download", e)
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            _model = Wav2Vec2Model.from_pretrained(model_name)
        _model.eval()
    return _model


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


def encode_audio(audio_bytes: bytes) -> list[float]:
    """把音频编码为 768 维向量（Wav2Vec2 最后一层 hidden state mean-pooling）。"""
    if settings.use_real_audio:
        try:
            return _encode_real(audio_bytes)
        except Exception as e:  # pragma: no cover
            logger.warning("Real audio encoding failed, fallback to mock: %s", e)
    return _mock_embed(audio_bytes, _AUDIO_DIM)


def _encode_real(audio_bytes: bytes) -> list[float]:
    import soundfile as sf
    import torch

    _load_model()
    wav_path = _to_wav_16k(audio_bytes)
    try:
        audio, _ = sf.read(wav_path, dtype="float32")
        audio = np.squeeze(audio)
        inputs = _feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, T, 768)
        vec = last_hidden.mean(dim=1).squeeze(0)
        vec = vec / max(float(torch.norm(vec)), 1e-12)
        return vec.tolist()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def warmup() -> None:
    if settings.use_real_audio:
        try:
            _load_model()
            logger.info("Audio encoder model loaded: facebook/wav2vec2-base-960h")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load Wav2Vec2, fallback to mock: %s", e)
    else:
        logger.info("Audio encoder uses mock encoder (dim=%s)", _AUDIO_DIM)