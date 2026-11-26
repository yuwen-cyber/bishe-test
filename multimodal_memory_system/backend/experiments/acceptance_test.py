"""按导师验收清单逐项实测：存得住 / 放得出 / 查得到 / 查得快 / 查得准 / 筛选。

输出结果写 experiments/output/acceptance_test_result.json，便于写论文与答辩引用。
"""
import json
import pathlib
import statistics
import time
from datetime import datetime

import sys

import requests
from minio import Minio
from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

BASE = "http://127.0.0.1:8000/api"
OUT = pathlib.Path("experiments/output")
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# ---------- 1. 服务与数据底数 ----------
eng = create_engine(settings.database_url)
with eng.connect() as c:
    conv_n = c.execute(text("SELECT COUNT(*) FROM conversations")).scalar()
    seg_n = c.execute(text("SELECT COUNT(*) FROM segments")).scalar()
    emo_dist = c.execute(text("SELECT emotion, COUNT(*) n FROM segments GROUP BY emotion")).all()
res["db"] = {"conversations": conv_n, "segments": seg_n,
              "emotion_dist": {e: int(n) for e, n in emo_dist}}

mc = Minio("localhost:9000", "minioadmin", "minioadmin", secure=False)
objs = [o.object_name for o in mc.list_objects("multimodal", recursive=True)]
res["minio"] = {"objects": len(objs),
                "audio": sum(1 for o in objs if o.startswith("audio")),
                "video": sum(1 for o in objs if o.startswith("video"))}

# ---------- 2. 存得住 / 放得出：取一条真实记录回放 ----------
with eng.connect() as c:
    empty_n = c.execute(text(
        "SELECT COUNT(*) FROM segments WHERE asr_text IS NULL OR TRIM(asr_text)=''"
    )).scalar()
res["asr_empty"] = {"count": int(empty_n), "total": seg_n,
                    "ratio": round(empty_n / seg_n, 4) if seg_n else None}

with eng.connect() as c:
    row = c.execute(text(
        "SELECT s.id, c.video_object_key, c.audio_object_key, s.asr_text "
        "FROM segments s JOIN conversations c ON s.conversation_id=c.id "
        "WHERE s.is_demo=0 AND s.id < 700 "
        "AND s.asr_text IS NOT NULL AND TRIM(s.asr_text) <> '' "
        "ORDER BY s.id LIMIT 1"
    )).mappings().first()

kw = row["asr_text"].strip()[:8] if row and row["asr_text"] else "压力"
q = requests.get(f"{BASE}/baseline/search", params={"q": kw}, timeout=60)
q.raise_for_status()
vurl = aurl = None
if q.json()["results"]:
    vurl = q.json()["results"][0].get("video_url")
    aurl = q.json()["results"][0].get("audio_url")

play = {}
for name, url in (("video", vurl), ("audio", aurl)):
    if not url:
        play[name] = {"ok": False, "reason": "接口未返回 url"}
        continue
    t0 = time.perf_counter()
    r = requests.get(url, timeout=60, stream=True)
    body = r.raw.read(200000)
    dt = (time.perf_counter() - t0) * 1000
    play[name] = {"ok": r.status_code == 200 and len(body) > 0,
                  "status": r.status_code,
                  "content_type": r.headers.get("content-type"),
                  "first_chunk_bytes": len(body),
                  "latency_ms": round(dt, 1)}
res["playback"] = play

# ---------- 3. 查得到：多查询命中 ----------
queries = ["压力很大", "被人敷衍", "当众讲话紧张", "整理房间", "睡不着"]
hit = []
for qq in queries:
    r = requests.get(f"{BASE}/baseline/search", params={"q": qq}, timeout=60)
    items = r.json()["results"] if r.ok else []
    hit.append({"q": qq, "n": len(items), "top": items[0]["asr_text"][:30] if items else ""})
res["retrieval_hit"] = hit

# ---------- 4. 标签筛选：中性 / 喜悦 ----------
filt = {}
for emo in ("中性", "喜悦"):
    r = requests.get(f"{BASE}/baseline/search", params={"q": "片段", "emotions": emo}, timeout=60)
    items = r.json()["results"] if r.ok else []
    filt[emo] = {"status": r.status_code, "n": len(items),
                 "emotions": [i["emotion"] for i in items],
                 "all_match": all(i["emotion"] == emo for i in items) if items else False}
# 混合检索同样测一遍
for emo in ("中性", "喜悦"):
    r = requests.get(f"{BASE}/hybrid/search", params={"q": "压力", "emotions": emo}, timeout=60)
    items = r.json().get("results", []) if r.ok else []
    filt[emo + "_hybrid"] = {"status": r.status_code, "n": len(items),
                             "emotions": [i["emotion"] for i in items],
                             "all_match": all(i["emotion"] == emo for i in items) if items else False}
res["emotion_filter"] = filt

# ---------- 5. 时间筛选 ----------
with eng.connect() as c:
    rng = c.execute(text("SELECT MIN(start_time) mn, MAX(start_time) mx FROM segments")).mappings().first()
mid = rng["mn"] + (rng["mx"] - rng["mn"]) / 2
r = requests.get(f"{BASE}/baseline/search",
                 params={"q": "压力", "start": str(mid), "end": str(rng["mx"])}, timeout=60)
items = r.json().get("results", []) if r.ok else []
in_range = []
for i in items:
    ts = datetime.fromisoformat(i["timestamp"].replace("Z", "")) if i.get("timestamp") else None
    in_range.append(bool(ts and ts >= mid))
res["time_filter"] = {"status": r.status_code, "n": len(items), "all_in_range": all(in_range) if items else None,
                      "window": [str(mid), str(rng["mx"])]}

# ---------- 6. 查得快：四模式延迟 ----------
lat = {}
for mode, ep in (("baseline", "baseline/search"), ("advanced", "advanced/search"),
                 ("hybrid", "hybrid/search"), ("structured", "structured/search")):
    ts = []
    params = {"q": "压力很大"} if mode != "structured" else {"q": "压力很大"}
    fail = None
    for _ in range(8):
        t0 = time.perf_counter()
        r = requests.get(f"{BASE}/{ep}", params=params, timeout=60)
        ts.append((time.perf_counter() - t0) * 1000)
        if not r.ok:
            fail = f"{r.status_code}"
    ts.sort()
    lat[mode] = {"status": fail or "ok", "p50_ms": round(statistics.median(ts), 1),
                 "p95_ms": round(ts[int(len(ts) * 0.95) - 1], 1), "max_ms": round(ts[-1], 1)}
res["latency"] = lat

# ---------- 7. 边界与潜在 bug ----------
bugs = {}
cases = [
    ("空查询", {"q": ""}, "baseline/search"),
    ("单字查询", {"q": "嗯"}, "baseline/search"),
    ("不存在的情绪标签", {"q": "压力", "emotions": "开心"}, "baseline/search"),
    ("起止时间颠倒", {"q": "压力", "start": str(rng["mx"]), "end": str(rng["mn"])}, "baseline/search"),
    ("超长查询", {"q": "压力" * 200}, "baseline/search"),
    ("特殊字符", {"q": "'\"%_\\"}, "baseline/search"),
]
for name, params, ep in cases:
    try:
        r = requests.get(f"{BASE}/{ep}", params=params, timeout=60)
        body = r.text[:160]
        bugs[name] = {"status": r.status_code, "body": body}
    except Exception as e:
        bugs[name] = {"error": f"{type(e).__name__}: {e}"}
res["edge_cases"] = bugs

OUT.joinpath("acceptance_test_result.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

print(json.dumps(res, ensure_ascii=False, indent=1))
