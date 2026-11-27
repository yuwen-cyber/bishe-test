"""精确统计影片噪声 2033~2259 各行的图像向量来源（mock 哈希 vs 真实 CLIP）。
只算 mock（哈希便宜），不匹配 mock 的即视为真实向量行，避免 227 次 CLIP 编码。
"""
import json
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import image_encoder, milvus_client, storage

client = storage.get_client()
col = milvus_client._get_collection()

rows, offset = [], 0
while True:
    b = col.query(expr="segment_id >= 2033 and segment_id <= 2259",
                  output_fields=["segment_id", "video_key", "image_embedding"],
                  limit=50, offset=offset)
    if not b:
        break
    rows += b
    offset += len(b)


def cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


mock_ids, other_ids = [], []
for r in rows:
    vb = client.get_object("multimodal", r["video_key"]).read()
    c = cos(r["image_embedding"], image_encoder._mock_embed(vb, 512))
    (mock_ids if c > 0.95 else other_ids).append(int(r["segment_id"]))

print(f"影片噪声总行数 {len(rows)}")
print(f"  mock 哈希向量 {len(mock_ids)} 条")
print(f"  非 mock（疑似真实） {len(other_ids)} 条  ids={other_ids[:40]}")
pathlib.Path("experiments/output/film_vector_source.json").write_text(
    json.dumps({"total": len(rows), "mock": mock_ids, "non_mock": other_ids}, ensure_ascii=False, indent=1),
    encoding="utf-8")
