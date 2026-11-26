"""针对标签筛选与时间筛选做可判定的验证：同一查询词，切换筛选条件，检查返回集合是否随之变化。"""
import json
import pathlib
import sys
from datetime import datetime, timedelta

import requests
from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

BASE = "http://127.0.0.1:8000/api"
eng = create_engine(settings.database_url)
out = {}


def call(params, ep="baseline/search"):
    r = requests.get(f"{BASE}/{ep}", params=params, timeout=60)
    return r.status_code, (r.json().get("results", []) if r.ok else [])


# ---------- 标签筛选：同一查询词，切换情绪 ----------
tag = {}
for ep in ("baseline/search", "hybrid/search", "structured/search"):
    for emo in ("中性", "喜悦", "悲伤"):
        st, items = call({"q": "我", "emotions": emo}, ep)
        tag[f"{ep}::{emo}"] = {
            "status": st, "n": len(items),
            "emotions": sorted({i["emotion"] for i in items}),
            "all_match": all(i["emotion"] == emo for i in items) if items else None,
        }
out["tag_filter"] = tag

# ---------- 时间筛选：同一查询词，切窗口 ----------
with eng.connect() as c:
    mx = c.execute(text("SELECT MAX(start_time) m FROM segments")).scalar()
    mn = c.execute(text("SELECT MIN(start_time) m FROM segments")).scalar()

wide = {"q": "我", "start": str(mn), "end": str(mx)}
st_all, items_all = call(wide)
narrow_end = mn + timedelta(seconds=(mx - mn).total_seconds() / 3)
st_narrow, items_narrow = call({"q": "我", "start": str(mn), "end": str(narrow_end)})
st_late, items_late = call({"q": "我", "start": str(mx - timedelta(seconds=3600)), "end": str(mx)})

def check(items, lo, hi):
    ok = []
    for i in items:
        ts = datetime.fromisoformat(i["timestamp"])
        ok.append(lo <= ts <= hi)
    return all(ok) if items else None

out["time_filter"] = {
    "full_range": {"n": len(items_all), "ids": [i["segment_id"] for i in items_all]},
    "first_third": {"n": len(items_narrow), "ids": [i["segment_id"] for i in items_narrow],
                    "window": [str(mn), str(narrow_end)],
                    "all_in_range": check(items_narrow, mn, narrow_end)},
    "last_hour": {"n": len(items_late), "ids": [i["segment_id"] for i in items_late],
                  "all_in_range": check(items_late, mx - timedelta(seconds=3600), mx)},
    "range_changes_results": [i["segment_id"] for i in items_all] != [i["segment_id"] for i in items_narrow],
}

# ---------- 组合筛选：情绪 + 时间 ----------
st, items = call({"q": "我", "emotions": "喜悦", "start": str(mn), "end": str(narrow_end)})
out["combined"] = {"status": st, "n": len(items),
                   "emotions": sorted({i["emotion"] for i in items}),
                   "ids": [i["segment_id"] for i in items]}

print(json.dumps(out, ensure_ascii=False, indent=1))
pathlib.Path("experiments/output/filter_test_result.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
