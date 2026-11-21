# 给 Trae 的话术：新增「演示面板」并移除「用户实验」

> 用法：整段复制粘贴到 Trae 对话框。前提：你已经在 
>
> `frontend`
>
>  目录下。
> 我已经把演示面板要用的数据导出好了：
>
> `frontend/public/demo-data.json`
>
> （26KB，静态快照，不用连后端）。



***

## 一、直接粘这段给 Trae



```
【任务背景】

Vue3 + Vite + Element Plus 前端项目，路径 D:\biyesheji\multimodal\_memory\_system\frontend。

现有页面：对话录制(/)、检索对比(/search)、用户实验(/lab)。

本次要做两件事：

(1) 删掉「用户实验」入口与路由；

(2) 新增「演示面板」(/demo)，用静态数据把六项实验的结果图表集中展示，替代散落在论文里的截图。

【硬约束，不要违反】

\- 只改 frontend 目录，绝对不要动 backend 任何文件。

\- 不要改 src/views/Home.vue、src/views/SearchCompare.vue、src/api/index.ts 的业务逻辑。

\- 不要引入 Element Plus 之外的新 UI 库；配色沿用现有白底 + #303133 文字 + #409eff 主色，禁止紫色渐变主题。

\- 演示面板必须能在后端完全不启动时正常打开（纯静态 JSON，不发任何 API 请求）。

【任务 A：移除用户实验】

1\. src/App.vue：

&#x20;  \- 删除菜单项 \<el-menu-item index="/lab">用户实验\</el-menu-item>

&#x20;  \- 新增菜单项 \<el-menu-item index="/demo">演示面板\</el-menu-item>，放在「检索对比」之后

2\. src/router/index.ts：

&#x20;  \- 删除 import ExperimentLab from '../views/ExperimentLab.vue' 及 /lab 路由

&#x20;  \- 新增 import DemoPanel from '../views/DemoPanel.vue' 与 { path: '/demo', component: DemoPanel }

3\. src/views/ExperimentLab.vue：

&#x20;  \- 先复制一份到 src/\_archive/ExperimentLab.vue.bak（目录不存在就创建），然后删除原文件

&#x20;  \- 这样论文/答辩要用到时还能找回，且不会因为残留 import 导致 vue-tsc 报错

4\. 确认全项目已无 "/lab" 与 "ExperimentLab" 的引用（用全局搜索确认）

【任务 B：安装依赖】

在 frontend 目录执行：npm install echarts

（如果 npm 安装失败或超时，不要卡住：改为用纯 CSS 画条形图实现同样的对比效果，并在页面底部注明「CSS 条形图降级版」）

【任务 C：新建 src/components/ChartBox.vue（图表容器）】

要求：

\- props: option: object, height?: string（默认 '340px'）

\- 内部用 echarts.init(ref) 渲染，watch(option) 时 setOption(option, true)

\- onMounted 后第一次 init；onUnmounted 必须 dispose

\- 监听 window resize 调 chart.resize()，onUnmounted 时移除监听

\- 不要每个页面重复写 init/dispose 逻辑，所有图表统一走这个组件

【任务 D：新建 src/views/DemoPanel.vue】

数据来源：fetch(\`\${import.meta.env.BASE\_URL}demo-data.json\`)，onMounted 里加载一次；

加载失败时显示 el-alert（type=error）：「未找到演示数据，请先在 backend 目录运行 experiments/export\_demo\_data.py」，且页面其余部分不崩。

数据结构（字段名必须严格照用，单位已说明）：

{

&#x20; meta: { generated\_at, project, data\_scale, notes: string\[] },

&#x20; sources: { \[key]: { file, desc, rows } },

&#x20; fourModes: \[ { config, "recall@1","recall@5","recall@10","precision@10","f1@10", mrr, "ndcg@10", avg\_time\_ms, client\_p50\_ms, client\_p95\_ms, num\_queries } ],   // 4 行: baseline / advanced\_plain / advanced\_lexicon / advanced\_llm

&#x20; hybrid:    \[ { config, "recall@1","recall@5","recall@10", mrr, "ndcg@10", avg\_time\_ms, p50\_time\_ms, p95\_time\_ms, num\_queries } ],                              // 4 行: keyword / vector / hybrid / hybrid\_weighted

&#x20; modalAblation:  \[ { config, "recall@1","recall@5","recall@10", mrr, "ndcg@10", avg\_time\_ms } ],   // 7 行: text\_only / audio\_only / image\_only / text\_audio / text\_image / text\_audio\_image / full\_plus\_emotion

&#x20; fusionAblation: \[ { strategy, "recall@5","recall@10", mrr, "ndcg@10", avg\_time\_ms } ],            // 3 行: concat\_pca / weighted\_avg / text\_only

&#x20; scale:     \[ { level, total\_avg\_ms, total\_p50\_ms, total\_p95\_ms, embed\_avg\_ms, search\_avg\_ms, mem\_mb, mem\_peak\_mb, cpu\_delta\_s } ],  // 3 行: 100 / 500 / 1000

&#x20; structured:\[ { mode, precision\_set, recall\_set, f1\_set, "precision@10","recall@10", mrr, p95\_time\_ms, num\_queries } ],  // 3 行: sql\_exact / keyword\_like / vector\_only

&#x20; byType:    \[ { config, type, n, "recall@5","recall@10", mrr, "ndcg@10" } ],   // 32 行，type 为四种中文：结构化精确查询 / 关键词查询 / 语义查询 / 跨模态查询

&#x20; cases:     \[ { query\_id, type, query, relevant\_ids, vector\_hits, keyword\_hits, baseline\_hits } ],  // 5 行

&#x20; overallAdjusted: \[ { source, config, scope, num\_queries, "recall@10", mrr, "ndcg@10" } ]           // 16 行，scope 为 原口径(50条) / 有效口径(38条)

}

页面结构（顶部用 el-tabs 分 4 个页签，避免一屏塞太满）：

页签 1「检索效果」：

\- 4 个 el-card 数字卡：数据规模（meta.data\_scale）、四模式最优 Recall@10、混合最优 Recall@10、有效口径最优 R@10（从 overallAdjusted 取 scope 含「有效」的最大值）

\- 图表 1：四模式对比（x=config；y 左轴 recall@10 与 mrr；y 右轴 client\_p95\_ms）——用双 y 轴，单位分别写「比例」与「ms」

\- 图表 2：混合检索对比（x=config；系列 recall@10、mrr）

\- 图表 3：按查询类型分组的热力图（x=四种 type，y=config，值为 recall@10），颜色深浅表达高低，tooltip 显示数值与 n

页签 2「消融实验」：

\- 图表 4：模态消融（x=7 个 config；系列 recall@10、mrr），config 名称在页面上替换成中文label：仅文本/仅音频/仅图像/文+音/文+图/文+音+图/全模态+情绪

\- 图表 5：融合策略消融（x=concat\_pca / weighted\_avg / text\_only；系列 recall@10、mrr）

\- 每张图下方一行小字「来源：metrics\_ablation.csv / metrics\_fusion\_ablation.csv」

页签 3「性能与规模」：

\- 图表 6：三档规模（x=100/500/1000，字符串类目；y 左 total\_p95\_ms，y 右 mem\_mb）

\- 图表 7：延迟分解（x=100/500/1000；堆叠柱：embed\_avg\_ms / search\_avg\_ms，说明「查询编码 vs 向量检索」占比）

\- 一张 el-table 展示三档的 cpu\_delta\_s、mem\_peak\_mb、total\_avg\_ms

页签 4「结构化查询与案例」：

\- 图表 8：结构化查询三方案（x=mode；系列 precision\_set、recall\_set、f1\_set），mode 中文label 精确过滤/纯关键词/纯向量

\- el-table 展示 cases：查询、类型、真值片段、向量命中、关键词命中、基线命中（命中字段是 "403|407" 这类竖线分隔字符串，原样展示，不要拆成数组）

\- 底部 el-alert（type=info，closable=false）逐条渲染 meta.notes（口径与局限），标题「口径说明」

【统一细节要求】

\- 所有比例数值展示为百分比，保留 2 位小数，例如 0.085 → 8.50%；延迟保留 1 位小数并带 ms

\- 每个图表容器固定高度（340px 或 300px），grid 必须 containLabel: true，避免轴标签被裁

\- 图表标题直接写中文，例如「四模式检索效果对比（50 条查询）」

\- 不要写死数据数组：图表数据全部从 demo-data.json 派生（map 出来），方便以后重跑实验替换 JSON

\- 图表 option 里不要出现 const 外部变量、不要用箭头函数做 formatter（保持简单，能不用 formatter 就不用）

【验收标准，逐条自查】

1\. frontend 目录执行 npx vue-tsc --noEmit 通过，无 TS 报错

2\. npm run build 通过

3\. 顶栏菜单为：对话录制 / 检索对比 / 演示面板；访问 /lab 不再有页面

4\. 后端完全不启动时，/demo 页面能正常渲染全部 8 张图与表格

5\. 浏览器控制台无报错、无 warning（尤其是 echarts 重复 init 的警告）

6\. 窗口宽度调到 1200px 以下，图表不重叠、文字不被裁切

7\. 全局搜索确认无 ExperimentLab 残留引用

【完成后请汇报】

\- 改了哪些文件（逐个列出）

\- vue-tsc 与 build 的实际输出结果

\- 8 张图分别在哪个页签、数据取自 demo-data.json 的哪个字段
```



***

## 二、Trae 可能追问的问题，你直接照这个答



| Trae 会问                           | 你这样答                                                          |
| --------------------------------- | ------------------------------------------------------------- |
| 要不要连后端接口？                         | 不要。演示面板只读 `public/demo-data.json` 静态快照，后端没启动也要能看。             |
| ExperimentLab.vue 直接删还是留？         | 先备份到 `src/_archive/ExperimentLab.vue.bak` 再删。答辩可能要找回，但路由必须摘掉。 |
| echarts 安装超时怎么办？                  | 降级用 CSS 条形图，页面底部标明「CSS 条形图降级版」，不要因此卡住整个任务。                    |
| 图表数据要不要写死？                        | 不许写死，全部从 JSON map 出来；以后重跑实验只要替换 JSON。                         |
| 能不能顺手把 Home/SearchCompare 样式也改统一？ | 不要。本次只新增页面 + 移除用户实验，其他视图一行不动。                                 |



***

## 三、以后重跑实验后怎么更新面板

在 `backend` 目录执行一句即可，前端刷新就变：



```
cd D:\biyesheji\multimodal\_memory\_system\backend

.\\.venv\Scripts\python.exe experiments\export\_demo\_data.py
```

它会重新读 `experiments/output/` 下 9 个 CSV，覆盖写 `frontend/public/demo-data.json`。



***

## 四、当前数据里必须留意的一点（答辩会被问）



* `structured` 那组里 `keyword_like` 只有 **3 条**适用查询（其余查询没有关键词），

  面板上要标「适用查询数 3」，不要把它的 100% 召回当成 20 条的成绩。

* `sql_exact` 的 P/R/F1 全 1.0 属构造必然（真值就是过滤结果），图上要加脚注说明，

  不要写成算法更优。

* 演示面板展示的是**静态快照**，生成时间在 `meta.generated_at`，建议页面顶部把它显示出来。