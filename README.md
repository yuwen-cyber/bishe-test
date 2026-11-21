# 面向桌面情感陪伴机器人的多模态对话记忆与语义检索系统

本科毕业设计项目。给桌面情感陪伴机器人加一套「能记住、能找回」的记忆系统：
把一次对话拆成片段，同时保存文本、语音情绪、语音特征、画面关键帧，
再用向量检索 + 结构化过滤把用户想找的那段对话捞出来。

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | Flask/FastAPI（`app.main:app`）、SQLAlchemy |
| 关系库 | MySQL 8（`segments` / `conversations` 两张表） |
| 向量库 | Milvus（HNSW + COSINE，4 路向量：文本 512 / 音频 768 / 图像 512 / 情绪 64） |
| 对象存储 | MinIO（音视频原件） |
| ASR | Paraformer-zh + FSMN-VAD + CT-Punc |
| 文本情绪 | mDeBERTa-v3-base-mnli-xnli（词典 0.85 : NLI 0.15 融合） |
| 语音情绪 | emotion2vec_plus_large |
| 文本向量 | BAAI/bge-small-zh-v1.5（512 维） |
| 音频向量 | wav2vec2-base-960h（768 维） |
| 图像向量 | CLIP ViT-B/32（512 维） |
| 前端 | Vue 3 + TypeScript + Vite + Element Plus |

## 目录结构

```
biyesheji/
├── multimodal_memory_system/
│   ├── backend/
│   │   ├── app/                    # 业务代码（检索、编码、入库）
│   │   └── experiments/            # 实验脚本与指标产物
│   │       ├── output/             # metrics_*.csv / *.md、查询集、映射表
│   │       ├── noise/              # 噪声数据的计划与素材清单（不含媒体）
│   │       └── *.py                # 数据构造、评测、替换、校验脚本
│   └── frontend/
│       └── src/                    # 三个页面：对话录制 / 检索对比 / 演示面板
├── query_expander_module/          # 查询扩展独立模块
├── 答辩演示操作手册.md
└── 前端演示面板-Trae话术.md
```

媒体素材（人脸切片、噪声音视频、合成的 1300 条素材）体积过大，
未纳入版本库，可用 `backend/experiments/` 下的脚本按清单重新生成。

## 数据规模

- 正式数据 700 条：真实录音转写 + 情绪匹配的人脸切片
- 噪声数据 1300 条：TTS 合成对白 + 网络免费场景素材切片（7 类情绪各 100 + 纯干扰 600）
- 用户自录 3 条
- 合计 2003 条，Milvus 行数一致

## 实验内容

| 实验 | 说明 |
|---|---|
| 四模式检索 | baseline（SQL 全文）/ advanced（纯向量）/ 混合检索 |
| 模态消融 | text / audio / image / 组合 / 全模态 |
| 融合策略消融 | 拼接降维 vs 加权平均 |
| 结构化精确查询 | SQL 条件检索 vs 向量 |
| 性能三档 | 100 / 500 / 1000 条规模下的延迟与内存 |
| 噪声鲁棒性 | clean（仅正式）vs full（全库） |

指标产物在 `backend/experiments/output/`，`metrics_*.csv` 与同名 `.md` 可直接引用。

## 本地运行

```bash
# 1. 基础设施
docker start mm-etcd mm-minio mm-milvus   # 或 docker compose up -d

# 2. 后端
cd multimodal_memory_system/backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env                       # 按需修改连接参数
.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 前端
cd ../frontend
npm install
npm run dev                                # http://localhost:5173
```

首次启动会加载 4 个模型，内存占用约 4GB；内存不足时可在 `.env` 关闭
`USE_REAL_AUDIO` / `USE_REAL_IMAGE`（离线灌库阶段已算好向量，线上可只保留文本与情绪）。

## 说明

- 正式数据的**音频**为真实录音，**画面**为按情绪匹配截取的人脸片段，二者非同框录制；
- 噪声数据为合成对白 + 场景素材，与正式数据不同源；
- 噪声情绪为构造标注，正式数据为模型预测，两者口径在论文中分别说明。
