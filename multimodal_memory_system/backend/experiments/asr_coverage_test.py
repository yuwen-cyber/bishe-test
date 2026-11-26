"""ASR 内容完整性抽样：30 条随机片段，比对 ASR 识别字数与库中文本长度。"""
import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app import asr, storage
from app.database import SessionLocal

db = SessionLocal()
rows = db.execute(text(
    "SELECT s.id, c.audio_object_key a, s.asr_text, s.emotion, s.is_demo "
    "FROM segments s JOIN conversations c ON s.conversation_id=c.id "
    "WHERE s.asr_text IS NOT NULL ORDER BY s.id"
)).mappings().all()
db.close()

random.seed(20261126)
sample = random.sample(rows, 30)
client = storage.get_client()
asr.warmup()

out = []
for r in sample:
    audio = client.get_object("multimodal", r["a"]).read()
    got = asr.transcribe(audio) or ""
    db_len = len(r["asr_text"] or "")
    ratio = round(len(got) / db_len, 2) if db_len else None
    out.append({
        "segment_id": int(r["id"]), "is_demo": bool(r["is_demo"]),
        "db_len": db_len, "asr_len": len(got), "ratio": ratio,
        "db_text": (r["asr_text"] or "")[:36],
        "asr_text": got[:36],
        "audio_kb": round(len(audio) / 1024, 1),
    })
    print(f"seg {r['id']:5d} 库 {db_len:3d} 字 / 识别 {len(got):3d} 字 比值 {ratio}", flush=True)

ratios = [x["ratio"] for x in out if x["ratio"] is not None]
bad = [x for x in out if x["ratio"] is not None and x["ratio"] < 0.6]
empty = [x for x in out if x["asr_len"] == 0]

summary = {
    "n": len(out),
    "ratio_mean": round(statistics.mean(ratios), 3),
    "ratio_min": min(ratios), "ratio_max": max(ratios),
    "ratio_std": round(statistics.pstdev(ratios), 3),
    "below_0.6": len(bad),
    "empty": len(empty),
    "bad_cases": bad,
}
print("\n=== ASR 完整性 ===")
print(json.dumps(summary, ensure_ascii=False, indent=1))

pathlib.Path("experiments/output/asr_coverage_result.json").write_text(
    json.dumps({"summary": summary, "samples": out}, ensure_ascii=False, indent=1),
    encoding="utf-8")
