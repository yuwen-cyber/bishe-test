#!/usr/bin/env python3
"""把长视频切出的影片片段并入噪声计划：生成台词、TTS 音频、视频副本。

在 noise_plan.json 末尾追加 N 条（noise_id 接着现有最大值），属性为：
    group=纯干扰、emotion=中性（影片空镜本身无情绪语义，不硬贴情绪标签）、
    voice 五音色轮换、duration_sec 取切片实际时长、source_type=film。

媒体落到 noise/audio/noise_XXXX.mp3 与 noise/video/noise_XXXX.mp4，
命名与原有噪声一致，可直接被 batch_import_noise.py 复用入库。

幂等：已存在的音频/视频跳过；重复执行不会重复追加（按 source_type+film 文件去重）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\build_film_plan.py --dry-run
    .venv\\Scripts\\python.exe experiments\\build_film_plan.py
    .venv\\Scripts\\python.exe experiments\\build_film_plan.py --max 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
NOISE_DIR = EXPERIMENTS_DIR / "noise"
PLAN_PATH = NOISE_DIR / "noise_plan.json"
FILM_MAP = NOISE_DIR / "film_slice_map.json"
FILM_DIR = NOISE_DIR / "film"
AUDIO_DIR = NOISE_DIR / "audio"
VIDEO_DIR = NOISE_DIR / "video"
BACKUP_PATH = NOISE_DIR / "noise_plan_before_film.json"

VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaoyiNeural",
]

SUBJ = [
    "会议室", "仓库", "打印机", "门禁卡", "排班表", "报销单", "快递", "投影仪",
    "服务器", "空调", "茶水间", "停车位", "工位", "外呼名单", "培训材料",
    "考勤记录", "设备清单", "备件柜", "会议室白板", "订餐表", "巡检单",
    "物料编码", "对账单", "用车申请", "门锁", "窗帘", "储物柜", "值班表",
    "监控", "工牌", "工单", "采购单", "台账", "交接记录", "应急预案",
    "值班电话", "样品柜", "档案袋", "周报", "评审日程",
]

ACT = [
    "要重新核对一遍", "先放着明天再处理", "已经交到前台了", "记得贴好标签",
    "按上周的格式改", "别动等我确认", "挪到三楼那间", "提前半小时准备好",
    "再发一份给行政", "按流程走就行", "先登记再领", "和上个月的对一下",
    "按新的编号排", "下班前弄完", "让财务先签个字", "照着清单点一遍",
    "这周内要归档", "别用旧模板", "先备份再说", "留一份纸质版",
    "按区域分开放", "等通知再动", "顺手把记录补上", "按部门分开列",
    "先试一遍再推广", "有问题就当场改", "按颜色区分", "盘点完再入库",
    "每次都要签字", "照旧例办",
]

TTS_CONCURRENCY = 4


def gen_texts(n: int, seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        a = rng.choice(SUBJ)
        b = rng.choice(ACT)
        c = rng.choice(SUBJ)
        d = rng.choice(ACT)
        if c == a or d == b:
            continue
        text = f"{a}{b}，{c}{d}。"
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


async def tts_one(sem: asyncio.Semaphore, text: str, voice: str, dst: Path) -> bool:
    import edge_tts

    async with sem:
        try:
            await edge_tts.Communicate(text=text, voice=voice, rate="-5%").save(str(dst))
            return True
        except Exception as e:
            print(f"  [TTS失败] {dst.name}: {type(e).__name__}: {e}", file=sys.stderr)
            return False


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="只处理前 N 条")
    parser.add_argument("--dry-run", action="store_true", help="只生成计划，不做 TTS")
    args = parser.parse_args()

    if not FILM_MAP.exists():
        print(f"找不到 {FILM_MAP}，先跑 build_film_slices.py --write")
        return 1

    film = load_json(FILM_MAP, {})
    clips = film.get("clips", [])
    if not clips:
        print("映射里没有切片记录")
        return 1

    plan = load_json(PLAN_PATH, [])
    if not plan:
        print(f"找不到 {PLAN_PATH}")
        return 1

    if args.max:
        clips = clips[:args.max]

    existing_film = {p.get("film_file") for p in plan if p.get("source_type") == "film"}
    todo = [c for c in clips if c.get("file") and c["file"] not in existing_film]

    next_id = max(int(p["noise_id"]) for p in plan) + 1
    print(f"现有计划 {len(plan)} 条｜影片候选 {len(clips)} 条｜待追加 {len(todo)} 条"
          f"｜起始 noise_id={next_id}")

    if not todo:
        print("没有需要追加的条目（已全部并入）")
        return 0

    texts = gen_texts(len(todo))
    new_items: list[dict] = []
    for i, (clip, text) in enumerate(zip(todo, texts)):
        nid = next_id + i
        new_items.append({
            "noise_id": nid,
            "group": "纯干扰",
            "emotion": "中性",
            "text": text,
            "source_video": str(FILM_DIR / clip["file"]),
            "start_sec": 0.0,
            "duration_sec": round(float(clip["duration_sec"]), 2),
            "voice": VOICES[i % len(VOICES)],
            "source_type": "film",
            "film_file": clip["file"],
            "film_source": clip["source_name"],
            "film_start_sec": clip["start_sec"],
        })

    print("\n样例：")
    for it in new_items[:5]:
        print(f"  {it['noise_id']} | {it['duration_sec']}s | {it['voice'][8:-6]} | {it['text']}")

    lens = [len(x["text"]) for x in new_items]
    print(f"\n文本长度 min {min(lens)} / 平均 {sum(lens)/len(lens):.1f} / max {max(lens)} 字")

    if args.dry_run:
        print("\n[dry-run] 未写入计划，也未生成媒体")
        return 0

    if not BACKUP_PATH.exists():
        shutil.copy2(PLAN_PATH, BACKUP_PATH)
        print(f"原计划已备份 → {BACKUP_PATH.name}")

    plan.extend(new_items)
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"计划已追加：{len(plan)} 条 → {PLAN_PATH}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    copied = 0
    for it in new_items:
        src = Path(it["source_video"])
        dst = VIDEO_DIR / f"noise_{it['noise_id']:04d}.mp4"
        if not dst.exists():
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"  [缺视频] {src}", file=sys.stderr)
    print(f"视频副本 {copied} 个（耗时 {time.time() - t0:.1f}s）")

    todo_tts = [it for it in new_items
                if not (AUDIO_DIR / f"noise_{it['noise_id']:04d}.mp3").exists()]
    print(f"待生成 TTS {len(todo_tts)} 条（并发 {TTS_CONCURRENCY}）")

    async def run_tts() -> int:
        sem = asyncio.Semaphore(TTS_CONCURRENCY)
        tasks = [
            tts_one(sem, it["text"], it["voice"],
                    AUDIO_DIR / f"noise_{it['noise_id']:04d}.mp3")
            for it in todo_tts
        ]
        results = await asyncio.gather(*tasks)
        return sum(1 for r in results if r)

    ok = asyncio.run(run_tts())
    print(f"TTS 完成 {ok}/{len(todo_tts)} 条，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0 if ok == len(todo_tts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
