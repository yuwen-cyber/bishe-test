"""重建 227 条影片噪声（segment_id 2033~2259）的图像/音频向量，使其与主库同源（真实 CLIP / 真实音频编码器）。

背景：这批数据在 USE_REAL_IMAGE=false / USE_REAL_AUDIO=false 下导入，Milvus 里落的是 mock 哈希向量，
与其余 2003 条真实向量不在同一空间（分层审计：影片噪声 10/10 为 mock，cos ≈ -0.05 vs 真实 CLIP）。

默认只做 dry-run（打印将要重建的行与相似度），加 --apply 才真正写入。
--apply 会先把原向量快照到 experiments/output/film_vector_rollback.json，便于回滚。

用法：
  .venv\\Scripts\\python.exe experiments\\rebuild_film_vectors.py            # 预览
  .venv\\Scripts\\python.exe experiments\\rebuild_film_vectors.py --apply    # 执行
"""
import argparse
import json
import pathlib
import sys
import time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import audio_encoder, image_encoder, milvus_client, storage
from app.config import settings

LO, HI = 2033, 2259
BATCH = 20
OUT_DIR = pathlib.Path(__file__).resolve().parent / "output"


def cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fetch_all(col, lo, hi):
    rows, offset = [], 0
    while True:
        batch = col.query(expr=f"segment_id >= {lo} and segment_id <= {hi}",
                          output_fields=["segment_id", "conversation_id", "text", "emotion",
                                         "start_time", "video_key", "audio_key", "embedding",
                                         "audio_embedding", "image_embedding", "emotion_embedding"],
                          limit=BATCH, offset=offset)
        if not batch:
            break
        rows += batch
        offset += len(batch)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写入 Milvus（默认 dry-run）")
    ap.add_argument("--force-real", action="store_true",
                    help="绕过 .env 的 USE_REAL_* 开关，强制用真实编码器编码")
    args = ap.parse_args()

    if not args.force_real and not (settings.use_real_image and settings.use_real_audio):
        print(f"[预处理] .env 当前 USE_REAL_IMAGE={settings.use_real_image} "
              f"USE_REAL_AUDIO={settings.use_real_audio}；将强制使用真实编码器（_encode_real）。")

    client = storage.get_client()
    col = milvus_client._get_collection()
    rows = fetch_all(col, LO, HI)
    print(f"取到影片噪声 {len(rows)} 条（segment_id {LO}~{HI}）")
    if not rows:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rebuilt, stats = [], {"img_was_mock": 0, "img_was_real": 0, "n": 0}
    for i, r in enumerate(rows, 1):
        vb = client.get_object("multimodal", r["video_key"]).read()
        ab = client.get_object("multimodal", r["audio_key"]).read()
        new_img = image_encoder._encode_real(vb)
        new_aud = audio_encoder._encode_real(ab)
        old_img_cos = cos(r["image_embedding"], new_img)
        stats["n"] += 1
        stats["img_was_real" if old_img_cos > 0.95 else "img_was_mock"] += 1
        rebuilt.append({
            "segment_id": int(r["segment_id"]),
            "conversation_id": int(r["conversation_id"]),
            "text": r["text"], "emotion": r["emotion"], "start_time": int(r["start_time"]),
            "video_key": r["video_key"], "audio_key": r["audio_key"],
            "embedding": list(r["embedding"]),
            "emotion_embedding": list(r["emotion_embedding"]),
            "old_image_embedding": list(r["image_embedding"]),
            "old_audio_embedding": list(r["audio_embedding"]),
            "new_image_embedding": new_img,
            "new_audio_embedding": new_aud,
        })
        if i % 20 == 0 or i == len(rows):
            print(f"  已编码 {i}/{len(rows)}  用时 {time.time() - t0:.0f}s")

    print(f"\n编码完成 {stats['n']} 条，其中原为 mock 向量 {stats['img_was_mock']} 条、"
          f"原为真实向量 {stats['img_was_real']} 条，总耗时 {time.time() - t0:.1f}s")

    if not args.apply:
        print("\n[dry-run] 未写入。确认无误后加 --apply 执行。")
        return

    snap = OUT_DIR / "film_vector_rollback.json"
    snap.write_text(json.dumps(rebuilt, ensure_ascii=False), encoding="utf-8")
    print(f"原向量快照已存：{snap}")

    ids = [x["segment_id"] for x in rebuilt]
    col.delete(f"segment_id in {ids}")
    print(f"已删除旧行 {len(ids)} 条，开始重新写入真实向量…")
    for x in rebuilt:
        milvus_client.insert_vector(
            segment_id=x["segment_id"], conversation_id=x["conversation_id"], text=x["text"],
            emotion=x["emotion"], start_time=x["start_time"],
            video_key=x["video_key"], audio_key=x["audio_key"],
            embedding=x["embedding"], audio_embedding=x["new_audio_embedding"],
            image_embedding=x["new_image_embedding"], emotion_embedding=x["emotion_embedding"],
        )
    print(f"写入完成，总耗时 {time.time() - t0:.1f}s")

    # 自检：抽 3 条核对库内向量已与真实编码器一致
    chk = fetch_all(col, LO, HI)
    print(f"\n[自检] 当前影片噪声行数 {len(chk)}")
    for r in chk[:3]:
        vb = client.get_object("multimodal", r["video_key"]).read()
        print(f"  seg {r['segment_id']}: cos(库, 真实CLIP) = "
              f"{cos(r['image_embedding'], image_encoder._encode_real(vb)):.3f}")


if __name__ == "__main__":
    main()
