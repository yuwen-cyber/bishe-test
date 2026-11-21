#!/usr/bin/env python3
"""构建 1300 条噪声数据（影视对白 + 纯干扰），用于把数据集从 700 扩充到 2000。

设计（与用户确认的口径）：
    总数 2000 = 正式 700（保留不动） + 噪声 1300
    噪声 1300 = 7 类情绪各 100 条（情绪噪声 700） + 纯干扰 600

每条噪声记录字段：
    noise_id        0~1299
    group           "喜悦"/"愤怒"/.../ 或 "纯干扰"
    emotion         情绪标签（纯干扰为 "中性"）
    text            台词（TTS 合成用）
    source_video    源视频绝对路径（从 face_videos 里选）
    start_sec       切片起点（秒）
    duration_sec    切片时长（秒）
    voice           音色

输出：experiments/noise/noise_plan.json
用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\build_noise_dataset.py
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
FACE_DIR = Path(r"D:\biyesheji\face_videos")
OUT_DIR = EXPERIMENTS_DIR / "noise"
PLAN_PATH = OUT_DIR / "noise_plan.json"

SEED = 42
PER_EMOTION = 100          # 7 类情绪各 100 条
PURE_NOISE = 600           # 纯干扰 600 条
CLIP_SEC = 10.0            # 目标切片时长
CLIP_JITTER = 2.0          # 时长抖动 ±2 秒

# face_videos 的英文目录名 → 中文情绪标签
VIDEO_PREFIX_TO_EMOTION = {
    "happy": "喜悦",
    "angry": "愤怒",
    "sad": "悲伤",
    "scared": "恐惧",
    "surprised": "惊讶",
    "disgusted": "厌恶",
    "expressionless": "中性",
}

# 情绪 → edge-tts 音色池（多音色轮换，模拟不同说话人）
VOICE_POOL = {
    "喜悦": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural"],
    "愤怒": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural", "zh-CN-YunyangNeural"],
    "悲伤": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunyangNeural", "zh-CN-XiaoyiNeural"],
    "恐惧": ["zh-CN-YunjianNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural"],
    "惊讶": ["zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"],
    "厌恶": ["zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural"],
    "中性": ["zh-CN-YunyangNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural"],
}

# --------------------------------------------------------------------------
# 情绪台词基础句（影视对白风格，内容与被试记忆无关）
# --------------------------------------------------------------------------
EMOTION_LINES = {
    "喜悦": [
        "成了！等了三年，终于成了。",
        "你看，我就说能行吧。",
        "今天这顿饭我请，谁也别跟我抢。",
        "她答应了，真的答应了。",
        "面试过了，下周一就能上班。",
        "票买到了，我们终于可以去看那场演出。",
        "医生说手术很成功，你放心吧。",
        "孩子第一次自己站起来走了两步。",
        "雨停了，太阳出来了，你看外面。",
        "这个月的活儿全部干完了，可以歇口气了。",
        "他给我回了消息，说他也一直在等。",
        "第一笔工资到账了，我想给我妈买件衣服。",
        "考试通过了，名单上有我的名字。",
        "车子终于修好了，明天就能上路。",
        "我们赢了，最后一球是我投进去的。",
        "房子定下来了，下周就能搬进去。",
        "她笑着说：你终于来了。",
        "这首歌我练了半年，今天一次唱完没出错。",
        "我把那封压了很久的信寄出去了，心里松了。",
        "花开了一片，你猜这是什么季节。",
        "爸，我拿到毕业证了。",
        "刚才那道题我做出来了，是真的做出来了。",
        "他答应把店盘给我，条件谈妥了。",
        "小张来了，他说愿意跟我们一块干。",
        "明天不上班，我们去看海。",
    ],
    "愤怒": [
        "你凭什么替我做这个决定？",
        "这件事我忍了很久了，今天必须说清楚。",
        "明明是你弄错了，还赖到我头上。",
        "说好了一起承担，结果全推给我一个人。",
        "我当时就站在那儿，一句话都说不出来。",
        "他答应过的事，一件都没做到。",
        "你们把人当什么了，说走就走。",
        "这钱是我一分一分攒的，凭什么给你拿走。",
        "话都说到这份上了，你还装听不见。",
        "昨天的事我一句都不信，你别再编了。",
        "我在外面站了三个小时，等到这句话。",
        "那份文件我早就交了，你们现在说没收到。",
        "别再拿这些理由糊弄我。",
        "这事要是这么办，我现在就把桌子拍了。",
        "你把我的东西挪哪去了，给我说清楚。",
        "他抢了我的功劳，还笑着说谢谢我配合。",
        "这么多年，我就没见过这么不讲理的。",
        "你要是早说，我也不至于走到今天这一步。",
        "一个人做错了还能理直气壮，真是服了。",
        "凭什么我来收拾这个烂摊子。",
        "我不是不讲道理的人，可你这么做实在过分。",
        "那次吵架之后，我再没跟他说过一句话。",
        "你把它退回去，我不接受这个处理结果。",
        "别再打这个电话了，我不想听。",
        "你心虚什么，说话都不敢看我。",
    ],
    "悲伤": [
        "她走的那天，我一句话都没说上。",
        "我妈走的时候，我还在外地。",
        "这封信我写了三年，一直没敢寄出去。",
        "我们一起走过了那么多年，最后一面都没见到。",
        "我站在门口，灯已经关了，屋里没人。",
        "搬家那天，我坐在地上愣了半个钟头。",
        "照片我留着，一直没敢放在桌上。",
        "他把钥匙放在桌上就走了，门轻轻带上。",
        "我做梦老是梦见那条老街，醒来就睡不着。",
        "那天雨特别大，我在站台上一直等，车没回来。",
        "我不知道该跟谁说，也不知道从哪儿说起。",
        "我以为我早就过去了，可今天又想起他。",
        "电话打过去是空号，我按了三次。",
        "最后一次见，他还在笑着说下次一起吃饭。",
        "家里的味道淡了，好像人真的走了。",
        "我一直记得那天的广播声，说车已经到站。",
        "有些话说晚了，就再也没机会说。",
        "我把那件衣服收起来了，放最上面的柜子里。",
        "这半年我没哭过，今天不知道为什么就哭了。",
        "我们就这样结束了，连一句好好告别都没有。",
        "每次路过那条街，我都会停一下。",
        "他是唯一一个认真听我说话的人。",
        "有些夜晚特别长，怎么也熬不到天亮。",
        "我把他的号码删了，又加回来，反复好几次。",
        "我觉得这一阵阵的空，是补不上的。",
    ],
    "恐惧": [
        "别出声，他们在楼下。",
        "那道门后面好像有动静，你听见了吗。",
        "别回头，往前走，一直往前走。",
        "灯怎么全灭了，谁在那儿。",
        "我数到十，要是没人回答我们就走。",
        "外面有人一直在敲门，敲了半个钟头。",
        "这地方不对劲，我们得现在就走。",
        "你把东西都收好，别留在桌上。",
        "我总觉得有人在后面跟着我，走了三条街了。",
        "电话那边一直没有声音，然后断了。",
        "我手一直在抖，钥匙都插不进去。",
        "别碰那个箱子，上面有标记。",
        "你说会不会有人知道了。",
        "这水越来越深了，好像没到底。",
        "我们不能再等，天黑之前必须过桥。",
        "刚才那个影子往这边看了，我看得很清楚。",
        "你听，楼梯上有人在走，一步一顿。",
        "门缝下面有光，说明屋里有人。",
        "我梦见那些东西了，醒来一身冷汗。",
        "车停在路边，一直没熄火，我不敢下车。",
        "这条路只有我们一辆车，后面一直有灯。",
        "我把它藏在砖底下了，你别问。",
        "别打开这个信封，真的别打开。",
        "他说话的语气变了，我在旁边不敢动。",
        "再等三分钟，要是没信号我们就撤。",
    ],
    "惊讶": [
        "你怎么会在这儿？",
        "不可能，昨天我还看见他在街口。",
        "这不是我写的，这字我一笔都没动过。",
        "等一下，你说的是我知道的那个名字？",
        "你把它打开了？现在？",
        "门口站着一个人，我认出来了。",
        "刚才还很吵，怎么一下就这么安静。",
        "他什么时候回来的，没人跟我说。",
        "这上面写的是昨天的日期。",
        "你居然把整箱都搬来了。",
        "我看了一遍，又看了一遍，还是不敢信。",
        "你说什么，再说一次。",
        "这么远的距离，他自己走过来的？",
        "桌上多了一个杯子，还是热的。",
        "我翻到最后一页，发现后面还有。",
        "灯亮着，门开着，人却不在了。",
        "这钱我一分没动，怎么会少了两张。",
        "他抬起头，我才认出是他。",
        "你手上这个东西，从哪弄来的。",
        "原来你一直都在，只是没出声。",
        "这车不是早上停在那儿的吗。",
        "他说他认识我，可我完全没印象。",
        "我把它拆开，里面是空的。",
        "你再说一遍，时间提前了？",
        "那不是昨天的事吗，怎么会是今天。",
    ],
    "厌恶": [
        "这种人我见一次恶心一次。",
        "当面一套背后一套，真让人受不了。",
        "他吃东西那声音，我一听着就头皮发麻。",
        "那种话说出来，也不觉得脏。",
        "你看他那副样子，我一句话都不想跟他讲。",
        "这事儿办得太难看了。",
        "我最烦的就是这种装模作样的人。",
        "他拍着胸口保证，转头就推得干干净净。",
        "屋子里一股说不出来的味儿，我站不住。",
        "你别再往这边凑了，站远点。",
        "他笑得那么假，谁看不出来。",
        "这种手段我听着都嫌脏耳朵。",
        "桌上那堆东西放了两天，都发霉了。",
        "他把别人做的活儿，说是自己的。",
        "跟这种人打交道，一次就够了。",
        "那种做法，我这辈子都接受不了。",
        "我看见他往里面吐了一口，我就再没吃过那家的东西。",
        "这地方不能久待，我待一会儿就想走。",
        "他嘴上说着关心，眼睛里全是别的意思。",
        "这事儿一提起来我心里就堵。",
        "为了那点好处，连脸都不要了。",
        "他大概忘了，那天我是站在旁边的。",
        "这种场面我实在看不下去了。",
        "水槽里泡着好几天没洗的碗。",
        "他去打扰人的时候，居然还挺理直气壮。",
    ],
    "中性": [
        "车停在后门，钥匙在手套箱里。",
        "明天九点开会，材料今天就得到。",
        "这份文件签两处，最后一页也要签。",
        "从这儿过去大概二十分钟，路上不堵。",
        "东西先放仓库，下午三点有人来取。",
        "会议改到三层第二会议室。",
        "发票抬头记得写全称，别简写。",
        "这一批一共三十箱，两箱是样品。",
        "我先过去看场地，你随后到。",
        "表格里的第三列要重新核对一遍。",
        "电话线路检修，明天上午可能联系不上。",
        "钥匙一共三把，最上面那把是大门的。",
        "把所有灯关掉再走，窗也检查一下。",
        "资料按日期排，最新的放最上面。",
        "这个月的支出我已经记完，明天核对。",
        "货架第二层清空了，新的明天到。",
        "会议室的白板笔没有了，去借两支。",
        "这周的值班表贴在门后面。",
        "两份都打印了吗，一份留底。",
        "机器先别关，还在跑数据。",
        "我下午去一趟，回来再跟你确认。",
        "表格里的数字我改了，你再看一遍。",
        "外面在下雨，伞放在门边的架子上。",
        "先把清单列出来，别的等下再说。",
        "周一到周五都有人，周末只开半天。",
    ],
}

# 纯干扰：场景描述 / 指令 / 日常事务（模板组合）
PURE_TEMPLATES = [
    "外面{weather}，{place}{act}。",
    "{obj}先放{place}，{time}再搬。",
    "从{place}过去大概{num}分钟，{traffic}。",
    "{place}那边的灯坏了，{time}有人来修。",
    "记得把{obj}清点一遍，数字记在{place}的本子上。",
    "{time}开会，{place}已经订好了。",
    "{obj}还剩{num}箱，先别下新单。",
    "这条路{traffic}，绕{place}走会快一点。",
    "{place}的门禁卡找不到了，先用侧门。",
    "{obj}的型号不对，{time}换一批。",
    "天气{weather}，{place}那边的窗记得关。",
    "先把{obj}登记，{place}那边下午核对。",
    "这批货从{place}进来，等{num}天到。",
    "会议材料{num}份，放{place}桌上。",
    "{time}把{obj}装好，车在{place}等。",
    "{place}的地面很滑，走路小心。",
    "电源在{place}，插头是{obj}旁边那个。",
    "电梯{time}保养，那段时间走楼梯。",
    "把{obj}按{num}个一组分开，装箱。",
    "上午去{place}取件，带上门禁卡。",
]

WEATHER = ["下雨", "起风", "很闷", "转晴", "有点冷", "雾很大", "阳光很好", "下小雪"]
PLACE = ["东门", "后街", "三号楼", "仓库", "楼下", "B区", "南边车库", "老食堂", "西侧通道", "二楼走廊"]
ACT = ["没几个人", "排着队", "比较安静", "在施工", "车位空着", "灯还没开", "关门了", "人多"]
OBJ = ["这批箱子", "那两箱样品", "打印的材料", "旧设备", "备用的椅子", "线材", "登记表", "工具箱"]
TIME = ["下午两点", "明天上午", "周五之前", "中午前", "下班之后", "九点半", "最早那班车之前", "傍晚"]
NUM = ["五", "十", "十五", "二十", "三", "八", "三十", "六"]
TRAFFIC = ["不堵", "在修路", "车不少", "没什么车", "有点堵", "只能过一辆"]


def _variant_text(line: str, emotion: str, rep: int) -> str:
    """同一条基础台词生成 4 个变体，避免 100 条里出现完全相同的句子。

    中性类是事务性陈述，不适合「说真的」这类口语前缀，单独给一套。
    """
    if rep == 0:
        return line
    if emotion == "中性":
        if rep == 1:
            return '另外，' + line
        if rep == 2:
            return line + '这个先记一下。'
        return '对了，' + line
    if rep == 1:
        return '说真的，' + line
    if rep == 2:
        return line + '我到现在还这么想。'
    return '你还记得吗，' + line


def _load_videos() -> dict[str, list[dict]]:
    """按情绪前缀收集源视频及其时长。"""
    pool: dict[str, list[dict]] = {em: [] for em in VIDEO_PREFIX_TO_EMOTION.values()}
    for f in sorted(FACE_DIR.glob("*.mp4")):
        prefix = "".join(ch for ch in f.stem if not ch.isdigit())
        emotion = VIDEO_PREFIX_TO_EMOTION.get(prefix)
        if not emotion:
            continue
        dur = _probe_duration(f)
        if dur >= CLIP_SEC:
            pool[emotion].append({"path": str(f), "dur": dur})
    return pool


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _build_pure_lines(rng: random.Random, need: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(lines) < need and guard < need * 200:
        guard += 1
        tpl = rng.choice(PURE_TEMPLATES)
        s = tpl.format(
            weather=rng.choice(WEATHER), place=rng.choice(PLACE), act=rng.choice(ACT),
            obj=rng.choice(OBJ), time=rng.choice(TIME), num=rng.choice(NUM),
            traffic=rng.choice(TRAFFIC),
        )
        if s not in seen:
            seen.add(s)
            lines.append(s)
    return lines


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    pool = _load_videos()

    for em, items in pool.items():
        if not items:
            print(f"[警告] 情绪 {em} 没有可用源视频")

    plan: list[dict] = []
    counter = {"v": 0}

    def pick_video(emotion: str) -> tuple[str, float]:
        """轮换取该情绪池里的视频，随机起点。"""
        items = pool[emotion]
        idx = counter["v"] % len(items)
        counter["v"] += 1
        item = items[idx]
        dur = round(rng.uniform(CLIP_SEC - CLIP_JITTER, CLIP_SEC + CLIP_JITTER), 2)
        max_start = max(0.0, item["dur"] - dur - 0.1)
        start = round(rng.uniform(0.0, max_start), 2) if max_start > 0 else 0.0
        return item["path"], start

    # --- 1) 7 类情绪噪声，各 100 条 ---
    for emotion, base_lines in EMOTION_LINES.items():
        produced = 0
        rep = 0
        while produced < PER_EMOTION:
            for line in base_lines:
                if produced >= PER_EMOTION:
                    break
                # 第 2 轮起使用变体，避免完全重复
                text = _variant_text(line, emotion, rep)
                path, start = pick_video(emotion)
                plan.append({
                    "noise_id": len(plan),
                    "group": emotion,
                    "emotion": emotion,
                    "text": text,
                    "source_video": path,
                    "start_sec": start,
                    "duration_sec": CLIP_SEC,
                    "voice": VOICE_POOL[emotion][produced % len(VOICE_POOL[emotion])],
                })
                produced += 1
            rep += 1

    # --- 2) 纯干扰 600 条 ---
    pure_lines = _build_pure_lines(rng, PURE_NOISE)
    all_videos = [it for items in pool.values() for it in items]
    for text in pure_lines:
        item = all_videos[counter["v"] % len(all_videos)]
        counter["v"] += 1
        dur = round(rng.uniform(CLIP_SEC - CLIP_JITTER, CLIP_SEC + CLIP_JITTER), 2)
        max_start = max(0.0, item["dur"] - dur - 0.1)
        start = round(rng.uniform(0.0, max_start), 2) if max_start > 0 else 0.0
        plan.append({
            "noise_id": len(plan),
            "group": "纯干扰",
            "emotion": "中性",
            "text": text,
            "source_video": item["path"],
            "start_sec": start,
            "duration_sec": dur,
            "voice": VOICE_POOL["中性"][counter["v"] % len(VOICE_POOL["中性"])],
        })

    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    from collections import Counter
    print("总条数:", len(plan))
    print("分组:", dict(Counter(p["group"] for p in plan)))
    print("文本去重后:", len({p["text"] for p in plan}))
    print("源视频使用次数:", dict(Counter(Path(p["source_video"]).stem for p in plan)))
    print("输出:", PLAN_PATH)

    # 抽 3 条示例
    print("\n示例：")
    for p in (plan[0], plan[100], plan[-1]):
        print(f"  [{p['group']}] {p['text']}  ({Path(p['source_video']).name} @ {p['start_sec']}s, {p['duration_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
