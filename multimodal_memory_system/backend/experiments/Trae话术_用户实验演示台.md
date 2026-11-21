# 交给 Trae 的话术：用户记忆体验实验演示台（创新点五）

> 用途：把下面「==== 话术正文 ====」到「==== 话术结束 ====」之间的内容整段复制给 Trae。
> 约束：只新增独立实验页面与配置文件，
>
> **不改动现有后端业务代码，不改动现有 Home.vue/ SearchCompare.vue**
>
> 。

\==== 话术正文 ====

## 背景

这是一个毕业设计项目，前端是 Vue3 + TypeScript + Vite + Element Plus + vue-router 4，Element Plus 已在 `src/main.ts` 全局注册。

现在需要新增一个**独立的 "用户记忆体验实验演示台" 页面**，用于做被试内（within-subjects）用户实验，对比机器人在三种记忆策略下的对话体验差异。

## 硬性约束（务必遵守）



1. **不要修改后端任何代码**，本实验台使用内置的预设对话脚本和回应，不调用现有 `/api/upload`、`/api/baseline/search`、`/api/advanced/search` 接口。

2. **不要改动现有文件**：`src/main.ts`、`src/App.vue`、`src/views/Home.vue`、`src/views/SearchCompare.vue`、`src/api/index.ts` 都保持原样。

3. 只允许**新增**下面列出的文件；路由只在 `src/router/index.ts` 的 `routes` 数组里**追加一条**新记录，不改动已有两条路由。

4. 技术栈沿用现有：Vue3 `<script setup lang="ts">`、Element Plus 组件、vue-router。

5. 所有问卷数据先存在浏览器 `localStorage`，并支持一键导出 CSV；不连数据库。

## 需要新建的文件

### 1. `src/experiment/materials.ts`（实验材料与问卷配置）

导出以下内容：

**（1）第一段对话记录（被试先阅读的历史背景）**，字段 `firstDialogue: string`，内容：

> 上周你和机器人聊过：最近项目赶进度，连续加班，压力很大，晚上睡不好；你养了一只橘猫叫 “橘子”，每天回家看到它会放松一点；你不太喜欢同事聚餐那种热闹场合。

**（2）三种条件下机器人的回忆回应**，键为条件 id，值为机器人开场回应文本：



* `no_memory`（无记忆条件）：

> 你好呀，今天想聊点什么？最近过得怎么样？有什么我可以帮你的吗？



* `keyword`（关键词检索条件，只抓到零散字面、回应有偏差、未准确串联）：

> 你之前提到过加班和猫对吧？那只猫最近怎么样了？工作上的事你再跟我说说？



* `semantic`（语义检索条件，准确复述历史关键事件并自然表达）：

> 记得你上周说项目赶进度、连续加班压力大、睡不好；还养了只叫橘子的橘猫，回家看到它会放松些。今天是又加班了，还是想聊聊橘子？

**（3）问卷题目**，7 级 Likert（1 = 非常不同意，7 = 非常同意），按维度分组：



```
export interface SurveyQuestion { key: string; dimension: string; text: string }

export const QUESTIONS: SurveyQuestion\[] = \[

&#x20; { key: 'M1', dimension: 'memory',    text: '机器人记得我之前告诉过它的事情' },

&#x20; { key: 'M2', dimension: 'memory',    text: '机器人记得我说过的具体细节' },

&#x20; { key: 'U1', dimension: 'understand', text: '机器人理解我当时的心情' },

&#x20; { key: 'U2', dimension: 'understand', text: '机器人回应贴合我表达的需求' },

&#x20; { key: 'T1', dimension: 'trust',     text: '我觉得这个机器人值得信赖' },

&#x20; { key: 'T2', dimension: 'trust',     text: '我愿意继续跟它分享我的心事' },

&#x20; { key: 'C1', dimension: 'companion', text: '和它对话让我感到陪伴' },

&#x20; { key: 'C2', dimension: 'companion', text: '它让我觉得不那么孤单' },

&#x20; { key: 'O1', dimension: 'overall',   text: '总体上我对这次对话体验满意' },

]
```

**（4）三条件拉丁方顺序表**（3 条件被试内设计，6 种排列，按被试编号序号对 6 取模分配）：



```
export const LATIN\_SQUARE: string\[]\[] = \[

&#x20; \['no\_memory', 'keyword', 'semantic'],

&#x20; \['no\_memory', 'semantic', 'keyword'],

&#x20; \['keyword', 'no\_memory', 'semantic'],

&#x20; \['keyword', 'semantic', 'no\_memory'],

&#x20; \['semantic', 'no\_memory', 'keyword'],

&#x20; \['semantic', 'keyword', 'no\_memory'],

]

export const CONDITION\_LABEL: Record\<string, string> = {

&#x20; no\_memory: '无记忆',

&#x20; keyword: '关键词检索',

&#x20; semantic: '语义检索',

}
```

### 2. `src/views/ExperimentLab.vue`（实验台页面）

用 Element Plus 实现，流程分步骤（用 `el-steps` 展示进度）：

**步骤 1・登记被试**



* 输入被试编号（如 S01、S02…），必填。

* 根据编号序号（去掉字母后的数字，从 1 开始）对 6 取模，从 `LATIN_SQUARE` 取本被试的条件顺序，页面上只提示 “本次体验顺序已安排好”，**不要在进入对应条件前提前暴露后面是什么条件**。

**步骤 2・阅读历史对话记录**



* 用 `el-card` 展示 `firstDialogue`，配一个 “我已读完，进入体验” 按钮。

**步骤 3・逐个条件体验并填问卷**

按分配好的顺序，一次只呈现一个条件：



1. 顶部提示当前是第几个体验（如 “第 1 / 3 个体验”）。

2. 聊天气泡样式展示该条件的机器人回应文本（不要显示条件的真实名称，只显示 “机器人回应”）。

3. 回应下方展示该条件对应的问卷：每个题目一行，用 `el-rate` 或 `el-radio-group` 提供 1\~7 分选择；题目必须全部作答才能提交。

4. 提交后进入下一个条件；条件之间显示 “请休息 2\~3 分钟后再继续”，配一个 “休息结束，继续” 按钮。

**步骤 4・完成与导出**



* 三个条件都完成后显示完成提示。

* 页面下方用 `el-table` 展示已收集到的所有被试数据（读 localStorage）。

* 提供 “导出 CSV” 按钮，把 localStorage 里的全部数据下载为 `user_experiment.csv`。

### 3. `src/router/index.ts` 追加路由（只追加，不改已有）

在 `routes` 数组末尾追加：



```
{ path: '/experiment', component: () => import('../views/ExperimentLab.vue') },
```

## 数据存储与导出格式



* localStorage 键名：`user_experiment_records`，值为记录数组。

* 每条记录字段（长表，一个被试 × 一个条件一行）：



```
subject\_id, order\_sequence, condition, trial\_index,

M1, M2, U1, U2, T1, T2, C1, C2, O1, submitted\_at
```



* `order_sequence`：本被试的条件顺序字符串，如 `no_memory>keyword>semantic`。

* `condition`：当前条件 id（`no_memory` / `keyword` / `semantic`）。

* `trial_index`：当前是第几个条件（1/2/3）。

* 各题目字段为 1\~7 的整数；`submitted_at` 为 ISO 时间字符串。

* 导出 CSV 用 UTF-8（带 BOM，方便 Excel 直接打开不乱码），表头与上面字段一致。

## 验收标准



1. 访问 `/experiment` 能打开实验台，现有 `/` 和 `/search` 页面功能不受影响。

2. 同一被试按编号自动分配拉丁方顺序，三个条件各体验一次、各填一次问卷。

3. 问卷未全选时不能提交；数据刷新页面后不丢失（localStorage）。

4. 导出的 CSV 字段完整、可被 Excel 正常打开、中文不乱码。

5. 不发起任何后端网络请求。

\==== 话术结束 ====

## 你自己需要做的部分（Trae 替代不了）



1. **招募 10\~15 名被试**（同学即可），编号 S01…S15。

2. **主持实验**：按页面安排的顺序，让被试先读历史记录，再依次体验三个条件；条件之间提醒休息 2\~3 分钟；指导语保持一致，不暗示哪种更好。

3. **回收 CSV**：全部做完后在实验台页面点 “导出 CSV”，保存 `user_experiment.csv`。

4. **跑统计分析**：把 CSV 交给配套脚本 `experiments/analyze_user_experiment.py`（已另附），出各条件得分、配对检验和效应量。

## 实验设计要点（写论文时用）



* 设计类型：被试内设计，每个被试都经历三种条件。

* 三条件：无记忆（不提历史）、关键词检索（字面回忆、易偏差）、语义检索（准确自然回忆）。

* 顺序平衡：拉丁方排列抵消顺序效应。

* 量表维度：被记住感、被理解感、信任感、陪伴感 + 总体满意度，7 级 Likert。

* 统计：配对 t 检验或 Wilcoxon 符号秩检验（按正态性选择），报告 Cohen's d 效应量；核心对比是 “语义检索 vs 关键词检索”“语义检索 vs 无记忆”。