from pathlib import Path

BASE = Path(__file__).resolve().parent
SEARCH = BASE / "app" / "routers" / "search.py"
HYBRID = BASE / "app" / "routers" / "hybrid.py"

s = SEARCH.read_text(encoding="utf-8")
s = s.replace(
    "from .. import embeddings, emotion, milvus_client, query_expander, schemas, storage",
    "from .. import (data_source, embeddings, emotion, milvus_client,\n"
    "                query_expander, schemas, storage)",
)
s = s.replace(
    '        is_demo=bool(row["is_demo"]),',
    '        is_demo=bool(row["is_demo"]),\n'
    '        data_source=data_source.data_source(row["id"], bool(row["is_demo"])),',
)
s = s.replace(
    '                is_demo=demo_map.get(int(h["segment_id"]), False),',
    '                is_demo=demo_map.get(int(h["segment_id"]), False),\n'
    '                data_source=data_source.data_source(\n'
    '                    h["segment_id"], demo_map.get(int(h["segment_id"]))\n'
    '                ),',
)
SEARCH.write_text(s, encoding="utf-8")
print("search.py 中 data_source 出现次数:", s.count("data_source"))

t = HYBRID.read_text(encoding="utf-8")
t = t.replace(
    '                is_demo=demo_map.get(sid, m.get("is_demo", False)),',
    '                is_demo=demo_map.get(sid, m.get("is_demo", False)),\n'
    '                data_source=data_source.data_source(\n'
    '                    sid, demo_map.get(sid, m.get("is_demo"))\n'
    '                ),',
)
HYBRID.write_text(t, encoding="utf-8")
print("hybrid.py 中 data_source 出现次数:", t.count("data_source"))
