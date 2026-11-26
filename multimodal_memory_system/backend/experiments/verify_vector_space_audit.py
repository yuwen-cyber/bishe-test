"""分层抽样审计：库内图像/音频向量到底来自真实模型还是 mock 哈希编码器。"""
import json
import pathlib
import random
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app import audio_encoder, image_encoder, milvus_client, storage
from app.database import SessionLocal

N_PER_STRATUM = 10
STRATA = {"正式(1~700)": (1, 700), "场景噪声(701~2032)": (701, 2032), "影片噪声(2033~2259)": (2033, 2259)}

db = SessionLocal()
ids_all = [int(x) for x in db.execute(text("SELECT id FROM segments ORDER BY id")).scalars().all()]
db.close()

random.seed(20261126)
selected = []
for name, (lo, hi) in STRATA.items():
    pool = [i for i in ids_all if lo <= i <= hi]
    if pool:
        selected += [(name, i) for i in random.sample(pool, min(N_PER_STRATUM, len(pool)))]

client = storage.get_client()
col = milvus_client._get_collection()

def cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

rows = {int(r["segment_id"]): r for r in col.query(
    expr=f"segment_id in {[i for _, i in selected]}",
    output_fields=["segment_id", "image_embedding", "audio_embedding", "video_key", "audio_key"])}

rec = []
for stratum, sid in selected:
    r = rows.get(sid)
    if not r:
        continue
    vb = client.get_object("multimodal", r["video_key"]).read()
    ab = client.get_object("multimodal", r["audio_key"]).read()
    c_real = cos(r["image_embedding"], image_encoder._encode_real(vb))
    c_run = cos(r["image_embedding"], image_encoder.encode_video_keyframe(vb))
    a_run = cos(r["audio_embedding"], audio_encoder.encode_audio(ab))
    img_src = "real" if c_real > 0.95 else ("mock" if c_run > 0.95 else "unknown")
    aud_src = "run-encoder" if a_run > 0.95 else "unknown"
    rec.append({"stratum": stratum, "segment_id": sid,
                "img_cos_real": round(c_real, 3), "img_cos_runtime": round(c_run, 3),
                "img_source": img_src, "aud_cos_runtime": round(a_run, 3), "aud_source": aud_src})
    print(f"{stratum:18s} seg {sid:5d} 图像={img_src:7s} cos_real={c_real:+.3f} cos_run={c_run:+.3f} 音频={aud_src}")

summary = {}
for name in STRATA:
    sub = [x for x in rec if x["stratum"] == name]
    summary[name] = {"n": len(sub),
                     "image_real": sum(1 for x in sub if x["img_source"] == "real"),
                     "image_mock": sum(1 for x in sub if x["img_source"] == "mock"),
                     "image_unknown": sum(1 for x in sub if x["img_source"] == "unknown")}

print("\n=== 分层统计 ===")
print(json.dumps(summary, ensure_ascii=False, indent=1))
pathlib.Path("experiments/output/vector_space_audit.json").write_text(
    json.dumps({"summary": summary, "samples": rec}, ensure_ascii=False, indent=1), encoding="utf-8")
