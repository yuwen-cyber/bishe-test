"""核实库内向量与运行时编码器是否同源：
对比 Milvus 中已存的 image/audio 向量与当前编码器输出（真实 CLIP vs mock 哈希）的余弦相似度。
"""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import audio_encoder, image_encoder, milvus_client, storage
from app.config import settings

client = storage.get_client()
col = milvus_client._get_collection()
ids = [27, 700, 2033, 2100, 2259]

print(f"USE_REAL_IMAGE={settings.use_real_image} USE_REAL_AUDIO={settings.use_real_audio}")

def cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

rows = {int(r["segment_id"]): r for r in col.query(
    expr=f"segment_id in {ids}", output_fields=["segment_id", "image_embedding", "audio_embedding", "video_key", "audio_key"]
)}

for sid, r in rows.items():
    vb = client.get_object("multimodal", r["video_key"]).read()
    ab = client.get_object("multimodal", r["audio_key"]).read()
    real_img = image_encoder._encode_real(vb)
    run_img = image_encoder.encode_video_keyframe(vb)   # 运行时实际用的路径
    run_aud = audio_encoder.encode_audio(ab)
    print(f"\nseg {sid}")
    print(f"  库 vs 真实CLIP      cos = {cos(r['image_embedding'], real_img):.3f}")
    print(f"  库 vs 运行时编码器  cos = {cos(r['image_embedding'], run_img):.3f}   <- 上传新数据时写入的就是这一路")
    print(f"  库 vs 运行时音频    cos = {cos(r['audio_embedding'], run_aud):.3f}")
