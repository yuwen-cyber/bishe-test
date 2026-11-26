"""分段耗时复现（单组件单进程，避免多模型同时驻留内存导致崩溃）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\perf_breakdown.py --component asr
组件：download / upload / asr / emotion / text_emb / audio_emb / image_emb
结果累加到 experiments/output/perf_breakdown.json
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import SessionLocal

IDS = [27, 300, 700, 2000, 2100, 2200]
OUT = pathlib.Path("experiments/output/perf_breakdown.json")


def load_rows():
    db = SessionLocal()
    rows = db.execute(text(
        "SELECT s.id, c.audio_object_key a, c.video_object_key v, s.asr_text, s.emotion, c.duration_ms "
        f"FROM segments s JOIN conversations c ON s.conversation_id=c.id "
        f"WHERE s.id IN ({','.join(map(str, IDS))}) ORDER BY s.id"
    )).mappings().all()
    db.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True)
    args = ap.parse_args()
    comp = args.component

    from app import storage
    client = storage.get_client()
    rows = load_rows()

    data = {}
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("components", {})
    data.setdefault("samples", {})

    # 组件预热（单独记录冷启动）
    t_cold = time.perf_counter()
    if comp == "asr":
        from app import asr
        asr.warmup()
    elif comp == "emotion":
        from app import emotion
        emotion.warmup()
    elif comp == "text_emb":
        from app import embeddings
        embeddings.warmup()
    elif comp == "image_emb":
        from app import image_encoder
        image_encoder.warmup()
    elif comp == "audio_emb":
        from app import audio_encoder
        audio_encoder.warmup() if hasattr(audio_encoder, "warmup") else None
    cold = time.perf_counter() - t_cold
    data.setdefault("cold_start_s", {})[comp] = round(cold, 1)

    times, details = [], []
    for r in rows:
        sid = int(r["id"])
        audio = client.get_object("multimodal", r["a"]).read()
        video = client.get_object("multimodal", r["v"]).read()

        if comp == "download":
            t = time.perf_counter()
            client.get_object("multimodal", r["a"]).read()
            client.get_object("multimodal", r["v"]).read()
        elif comp == "upload":
            t = time.perf_counter()
            storage.upload_bytes(audio, f"_perf/{sid}_a.webm", "audio/webm")
            storage.upload_bytes(video, f"_perf/{sid}_v.webm", "video/webm")
            client.remove_object("multimodal", f"_perf/{sid}_a.webm")
            client.remove_object("multimodal", f"_perf/{sid}_v.webm")
        elif comp == "asr":
            from app import asr
            t = time.perf_counter()
            text_out = asr.transcribe(audio)
        elif comp == "emotion":
            from app import emotion
            t = time.perf_counter()
            emo, conf = emotion.predict(r["asr_text"] or "", audio)
        elif comp == "text_emb":
            from app import embeddings
            t = time.perf_counter()
            embeddings.embed_text(r["asr_text"] or "空")
        elif comp == "audio_emb":
            from app import audio_encoder
            t = time.perf_counter()
            audio_encoder.encode_audio(audio)
        elif comp == "image_emb":
            from app import image_encoder
            t = time.perf_counter()
            image_encoder._encode_real(video)
        else:
            raise SystemExit(f"未知组件 {comp}")
        dt = (time.perf_counter() - t) * 1000
        times.append(dt)

        rec = data["samples"].setdefault(str(sid), {
            "segment_id": sid, "duration_ms": r["duration_ms"],
            "db_text_len": len(r["asr_text"] or ""), "db_emotion": r["emotion"],
            "audio_kb": round(len(audio) / 1024, 1), "video_kb": round(len(video) / 1024, 1),
        })
        if comp == "asr":
            rec["asr_text_len"] = len(text_out or "")
            rec["asr_text"] = (text_out or "")[:40]
            rec["asr_ratio"] = round(len(text_out) / rec["db_text_len"], 2) if rec["db_text_len"] else None
        if comp == "emotion":
            rec["emotion_now"] = emo
            rec["emotion_conf"] = conf
        rec[f"t_{comp}_ms"] = round(dt, 1)
        details.append((sid, round(dt, 1), rec.get("asr_ratio")))

    data["components"][comp] = {
        "mean_ms": round(statistics.mean(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "std_ms": round(statistics.pstdev(times), 1),
        "n": len(times),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[{comp}] 均值 {data['components'][comp]['mean_ms']}ms "
          f"区间 {min(times):.1f}~{max(times):.1f}ms "
          f"标准差 {statistics.pstdev(times):.1f}ms 冷启动 {cold:.1f}s")
    for sid, ms, ratio in details:
        extra = f" ASR字数比={ratio}" if ratio is not None else ""
        print(f"  seg {sid}: {ms}ms{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
