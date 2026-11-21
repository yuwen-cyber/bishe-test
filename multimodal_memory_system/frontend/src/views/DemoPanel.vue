<template>
  <div class="demo-page">
    <el-alert
      v-if="loadError"
      type="error"
      :closable="false"
      show-icon
      :title="loadError"
      class="mb"
    />

    <template v-if="data">
      <div class="page-head">
        <h2 class="title">实验演示面板</h2>
        <p class="sub">{{ data.meta.project }}</p>
        <p class="sub">数据生成时间：{{ data.meta.generated_at }}</p>
      </div>

      <el-tabs v-model="activeTab" class="tabs">
        <!-- 页签 1：检索效果 -->
        <el-tab-pane label="检索效果" name="effect" lazy>
          <div class="stat-grid">
            <el-card shadow="never" class="stat-card">
              <div class="stat-label">数据规模</div>
              <div class="stat-value small">{{ cardScale }}</div>
            </el-card>
            <el-card shadow="never" class="stat-card">
              <div class="stat-label">四模式最优 Recall@10</div>
              <div class="stat-value">{{ cardBestFour }}</div>
            </el-card>
            <el-card shadow="never" class="stat-card">
              <div class="stat-label">混合最优 Recall@10</div>
              <div class="stat-value">{{ cardBestHybrid }}</div>
            </el-card>
            <el-card shadow="never" class="stat-card">
              <div class="stat-label">有效口径最优 Recall@10</div>
              <div class="stat-value">{{ cardBestAdjusted }}</div>
            </el-card>
          </div>

          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartFourModes" height="340px" />
          </el-card>

          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartHybrid" height="340px" />
          </el-card>

          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartHeatmap" height="360px" />
          </el-card>
        </el-tab-pane>

        <!-- 页签 2：消融实验 -->
        <el-tab-pane label="消融实验" name="ablation" lazy>
          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartModalAblation" height="340px" />
            <p class="source">来源：{{ sourceFile('modalAblation', 'metrics_ablation.csv') }}</p>
          </el-card>

          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartFusionAblation" height="340px" />
            <p class="source">来源：{{ sourceFile('fusionAblation', 'metrics_fusion_ablation.csv') }}</p>
          </el-card>
        </el-tab-pane>

        <!-- 页签 3：性能与规模 -->
        <el-tab-pane label="性能与规模" name="scale" lazy>
          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartScale" height="340px" />
          </el-card>

          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartLatency" height="340px" />
            <p class="source">查询编码（embed）与向量检索（search）耗时分解</p>
          </el-card>

          <el-card shadow="never" class="chart-card">
            <el-table :data="data.scale" size="small" border>
              <el-table-column prop="level" label="数据规模（条）" width="150" />
              <el-table-column label="总耗时均值">
                <template #default="{ row }">{{ fmtMs(row.total_avg_ms) }}</template>
              </el-table-column>
              <el-table-column label="CPU 增量（s）">
                <template #default="{ row }">{{ row.cpu_delta_s.toFixed(2) }}</template>
              </el-table-column>
              <el-table-column label="峰值内存（MB）">
                <template #default="{ row }">{{ row.mem_peak_mb.toFixed(1) }}</template>
              </el-table-column>
            </el-table>
          </el-card>
        </el-tab-pane>

        <!-- 页签 4：结构化查询与案例 -->
        <el-tab-pane label="结构化查询与案例" name="structured" lazy>
          <el-card shadow="never" class="chart-card">
            <ChartBox :option="chartStructured" height="340px" />
            <p class="caption">图中为 Top-10 排名口径。精确过滤的精确率@10 为 100%（返回项全部属于真值集），召回率@10 受真值规模限制而明显偏低（真值平均 75.6 条，前 10 条装不满）；集合级精确率/召回率/F1 的 100% 属构造必然，仅证明能力可用，不构成算法优劣比较。纯关键词仅 3 条查询适用（n=3），与另两组 20 条不可直接比较。</p>
          </el-card>

          <el-card shadow="never" class="chart-card">
            <el-table :data="data.cases" size="small" border>
              <el-table-column prop="query" label="查询" min-width="200" />
              <el-table-column prop="type" label="类型" width="110" />
              <el-table-column prop="relevant_ids" label="真值片段" min-width="120" />
              <el-table-column prop="vector_hits" label="向量命中" min-width="110" />
              <el-table-column prop="keyword_hits" label="关键词命中" min-width="110" />
              <el-table-column prop="baseline_hits" label="基线命中" min-width="110" />
            </el-table>
          </el-card>

          <el-alert type="info" :closable="false" title="口径说明" class="mb">
            <div v-for="(note, idx) in data.meta.notes" :key="idx" class="note-line">
              {{ idx + 1 }}. {{ note }}
            </div>
          </el-alert>
        </el-tab-pane>
      </el-tabs>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import ChartBox from '../components/ChartBox.vue'

// ---------- 数据类型（字段名与 demo-data.json 严格一致） ----------
interface Meta {
  generated_at: string
  project: string
  data_scale: string
  notes: string[]
}
interface SourceMeta {
  file: string
  desc: string
  rows: number
}
interface FourModeRow {
  config: string
  'recall@1': number
  'recall@5': number
  'recall@10': number
  'precision@10': number
  'f1@10': number
  mrr: number
  'ndcg@10': number
  avg_time_ms: number
  client_p50_ms: number
  client_p95_ms: number
  num_queries: number
}
interface HybridRow {
  config: string
  'recall@1': number
  'recall@5': number
  'recall@10': number
  mrr: number
  'ndcg@10': number
  avg_time_ms: number
  p50_time_ms: number
  p95_time_ms: number
  num_queries: number
}
interface ModalRow {
  config: string
  'recall@1': number
  'recall@5': number
  'recall@10': number
  mrr: number
  'ndcg@10': number
  avg_time_ms: number
}
interface FusionRow {
  strategy: string
  'recall@5': number
  'recall@10': number
  mrr: number
  'ndcg@10': number
  avg_time_ms: number
}
interface ScaleRow {
  level: number
  total_avg_ms: number
  total_p50_ms: number
  total_p95_ms: number
  embed_avg_ms: number
  search_avg_ms: number
  mem_mb: number
  mem_peak_mb: number
  cpu_delta_s: number
}
interface StructuredRow {
  mode: string
  precision_set: number
  recall_set: number
  f1_set: number
  'precision@10': number
  'recall@10': number
  'f1@10': number
  mrr: number
  p95_time_ms: number
  num_queries: number
}
interface ByTypeRow {
  config: string
  type: string
  n: number
  'recall@5': number
  'recall@10': number
  mrr: number
  'ndcg@10': number
}
interface CaseRow {
  query_id: number
  type: string
  query: string
  relevant_ids: string
  vector_hits: string
  keyword_hits: string
  baseline_hits: string
}
interface OverallRow {
  source: string
  config: string
  scope: string
  num_queries: number
  'recall@10': number
  mrr: number
  'ndcg@10': number
}
interface DemoData {
  meta: Meta
  sources: Record<string, SourceMeta>
  fourModes: FourModeRow[]
  hybrid: HybridRow[]
  modalAblation: ModalRow[]
  fusionAblation: FusionRow[]
  scale: ScaleRow[]
  structured: StructuredRow[]
  byType: ByTypeRow[]
  cases: CaseRow[]
  overallAdjusted: OverallRow[]
}

// ---------- 数据加载（纯静态 JSON，不发后端请求） ----------
const data = ref<DemoData | null>(null)
const loadError = ref('')
const activeTab = ref('effect')

onMounted(async () => {
  try {
    const resp = await fetch(`${import.meta.env.BASE_URL}demo-data.json`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    data.value = (await resp.json()) as DemoData
  } catch (e) {
    loadError.value = `未找到演示数据，请先在 backend 目录运行 experiments/export_demo_data.py（${String(e)}）`
  }
})

// ---------- 通用格式化 ----------
function pct(v: number | undefined): number {
  return Number((((v ?? 0) * 100)).toFixed(2))
}
function fmtPct(v: number | undefined): string {
  return `${((v ?? 0) * 100).toFixed(2)}%`
}
function fmtMs(v: number | undefined): string {
  return `${(v ?? 0).toFixed(1)} ms`
}
function sourceFile(key: string, fallback: string): string {
  return data.value?.sources?.[key]?.file ?? fallback
}

// ---------- 数字卡 ----------
const cardScale = computed(() => data.value?.meta.data_scale ?? '-')
const cardBestFour = computed(() => {
  const rows = data.value?.fourModes ?? []
  if (!rows.length) return '-'
  return fmtPct(Math.max(...rows.map((r) => r['recall@10'])))
})
const cardBestHybrid = computed(() => {
  const rows = data.value?.hybrid ?? []
  if (!rows.length) return '-'
  return fmtPct(Math.max(...rows.map((r) => r['recall@10'])))
})
const cardBestAdjusted = computed(() => {
  const rows = (data.value?.overallAdjusted ?? []).filter((r) => r.scope.includes('有效'))
  if (!rows.length) return '-'
  return fmtPct(Math.max(...rows.map((r) => r['recall@10'])))
})

// ---------- 图表 1：四模式对比（双 y 轴） ----------
const chartFourModes = computed(() => {
  const rows = data.value?.fourModes ?? []
  const n = rows[0]?.num_queries ?? 0
  return {
    title: {
      text: `四模式检索效果对比（${n} 条查询）`,
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((r) => r.config),
      axisLabel: { color: '#303133' },
    },
    yAxis: [
      {
        type: 'value',
        name: '比例（%）',
        axisLabel: { formatter: '{value}%', color: '#303133' },
        nameTextStyle: { color: '#303133' },
      },
      {
        type: 'value',
        name: '延迟（ms）',
        axisLabel: { formatter: '{value} ms', color: '#303133' },
        nameTextStyle: { color: '#303133' },
      },
    ],
    series: [
      {
        name: 'Recall@10（%）',
        type: 'bar',
        barMaxWidth: 36,
        data: rows.map((r) => pct(r['recall@10'])),
        itemStyle: { color: '#409eff' },
      },
      {
        name: 'MRR（%）',
        type: 'bar',
        barMaxWidth: 36,
        data: rows.map((r) => pct(r.mrr)),
        itemStyle: { color: '#79bbff' },
      },
      {
        name: 'P95 延迟（ms）',
        type: 'line',
        yAxisIndex: 1,
        data: rows.map((r) => Number(r.client_p95_ms.toFixed(1))),
        itemStyle: { color: '#e6a23c' },
      },
    ],
  }
})

// ---------- 图表 2：混合检索对比 ----------
const chartHybrid = computed(() => {
  const rows = data.value?.hybrid ?? []
  const n = rows[0]?.num_queries ?? 0
  return {
    title: {
      text: `混合检索方案对比（${n} 条查询）`,
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((r) => r.config),
      axisLabel: { color: '#303133' },
    },
    yAxis: {
      type: 'value',
      name: '比例（%）',
      axisLabel: { formatter: '{value}%', color: '#303133' },
      nameTextStyle: { color: '#303133' },
    },
    series: [
      {
        name: 'Recall@10（%）',
        type: 'bar',
        barMaxWidth: 40,
        data: rows.map((r) => pct(r['recall@10'])),
        itemStyle: { color: '#409eff' },
      },
      {
        name: 'MRR（%）',
        type: 'bar',
        barMaxWidth: 40,
        data: rows.map((r) => pct(r.mrr)),
        itemStyle: { color: '#79bbff' },
      },
    ],
  }
})

// ---------- 图表 3：按查询类型分组热力图 ----------
const HEAT_TYPES = ['结构化精确查询', '关键词查询', '语义查询', '跨模态查询']

const heatConfigs = computed(() => {
  const list = (data.value?.byType ?? []).map((r) => r.config)
  return Array.from(new Set(list))
})

const heatData = computed(() => {
  const configs = heatConfigs.value
  return (data.value?.byType ?? []).map((r) => [
    HEAT_TYPES.indexOf(r.type),
    configs.indexOf(r.config),
    pct(r['recall@10']),
    r.n,
  ])
})

const HEAT_LABELS: Record<string, string> = {
  baseline: '基准（全文）',
  advanced_plain: '向量（无扩展）',
  advanced_lexicon: '向量（词典扩展）',
  advanced_llm: '向量（LLM 扩展）',
  keyword: '混合-关键词',
  vector: '混合-向量',
  hybrid: '混合（RRF）',
  hybrid_weighted: '混合（加权）',
}

// 颜色上限按实际最大值取整，避免默认 0~60 让所有格子挤在同一档浅色
const heatMax = computed(() => {
  const values = heatData.value.map((d) => d[2] as number)
  return values.length ? Math.max(1, Math.ceil(Math.max(...values))) : 100
})

function heatTooltip(param: { data: number[] }): string {
  const [xIdx, yIdx, value, n] = param.data
  const raw = heatConfigs.value[yIdx] ?? ''
  const cfg = HEAT_LABELS[raw] ?? raw
  const type = HEAT_TYPES[xIdx] ?? ''
  return `${cfg}<br/>${type}<br/>Recall@10：${value}%<br/>查询数 n=${n}`
}

function heatLabel(param: { data: number[] }): string {
  return `${(param.data[2] as number).toFixed(1)}%`
}

const chartHeatmap = computed(() => ({
  title: {
    text: '按查询类型分组的 Recall@10 热力图',
    left: 'center',
    textStyle: { fontSize: 14, color: '#303133' },
  },
  tooltip: { formatter: heatTooltip },
  grid: { left: 10, right: 20, top: 70, bottom: 60, containLabel: true },
  xAxis: {
    type: 'category',
    data: HEAT_TYPES,
    axisLabel: { color: '#303133' },
    splitArea: { show: true },
  },
  yAxis: {
    type: 'category',
    data: heatConfigs.value.map((c) => HEAT_LABELS[c] ?? c),
    axisLabel: { color: '#303133' },
    splitArea: { show: true },
  },
  visualMap: {
    min: 0,
    max: heatMax.value,
    calculable: true,
    orient: 'horizontal',
    left: 'center',
    bottom: 0,
    text: ['高', '低'],
    textStyle: { color: '#303133' },
    inRange: { color: ['#f5f9ff', '#d6e8ff', '#9ecbff', '#5aa9f5', '#2f86e0'] },
  },
  series: [
    {
      name: 'Recall@10',
      type: 'heatmap',
      data: heatData.value,
      label: { show: true, formatter: heatLabel, fontSize: 11, color: '#303133' },
      emphasis: { itemStyle: { borderColor: '#303133', borderWidth: 1 } },
    },
  ],
}))

// ---------- 图表 4：模态消融 ----------
const MODAL_LABELS: Record<string, string> = {
  text_only: '仅文本',
  audio_only: '仅音频',
  image_only: '仅图像',
  text_audio: '文+音',
  text_image: '文+图',
  text_audio_image: '文+音+图',
  full_plus_emotion: '全模态+情绪',
}

const chartModalAblation = computed(() => {
  const rows = data.value?.modalAblation ?? []
  return {
    title: {
      text: '模态消融实验（7 组配置）',
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((r) => MODAL_LABELS[r.config] ?? r.config),
      axisLabel: { color: '#303133' },
    },
    yAxis: {
      type: 'value',
      name: '比例（%）',
      axisLabel: { formatter: '{value}%', color: '#303133' },
      nameTextStyle: { color: '#303133' },
    },
    series: [
      {
        name: 'Recall@10（%）',
        type: 'bar',
        barMaxWidth: 32,
        data: rows.map((r) => pct(r['recall@10'])),
        itemStyle: { color: '#409eff' },
      },
      {
        name: 'MRR（%）',
        type: 'bar',
        barMaxWidth: 32,
        data: rows.map((r) => pct(r.mrr)),
        itemStyle: { color: '#79bbff' },
      },
    ],
  }
})

// ---------- 图表 5：融合策略消融 ----------
const chartFusionAblation = computed(() => {
  const rows = data.value?.fusionAblation ?? []
  return {
    title: {
      text: '融合策略消融实验',
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((r) => r.strategy),
      axisLabel: { color: '#303133' },
    },
    yAxis: {
      type: 'value',
      name: '比例（%）',
      axisLabel: { formatter: '{value}%', color: '#303133' },
      nameTextStyle: { color: '#303133' },
    },
    series: [
      {
        name: 'Recall@10（%）',
        type: 'bar',
        barMaxWidth: 46,
        data: rows.map((r) => pct(r['recall@10'])),
        itemStyle: { color: '#409eff' },
      },
      {
        name: 'MRR（%）',
        type: 'bar',
        barMaxWidth: 46,
        data: rows.map((r) => pct(r.mrr)),
        itemStyle: { color: '#79bbff' },
      },
    ],
  }
})

// ---------- 图表 6：三档规模 ----------
const chartScale = computed(() => {
  const rows = data.value?.scale ?? []
  const levels = rows.map((r) => String(r.level))
  return {
    title: {
      text: '数据规模对检索性能的影响',
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: levels,
      axisLabel: { color: '#303133' },
    },
    yAxis: [
      {
        type: 'value',
        name: 'P95 延迟（ms）',
        axisLabel: { formatter: '{value} ms', color: '#303133' },
        nameTextStyle: { color: '#303133' },
      },
      {
        type: 'value',
        name: '内存（MB）',
        axisLabel: { formatter: '{value} MB', color: '#303133' },
        nameTextStyle: { color: '#303133' },
      },
    ],
    series: [
      {
        name: '总耗时 P95（ms）',
        type: 'bar',
        barMaxWidth: 50,
        data: rows.map((r) => Number(r.total_p95_ms.toFixed(1))),
        itemStyle: { color: '#409eff' },
      },
      {
        name: '内存占用（MB）',
        type: 'line',
        yAxisIndex: 1,
        data: rows.map((r) => Number(r.mem_mb.toFixed(1))),
        itemStyle: { color: '#e6a23c' },
      },
    ],
  }
})

// ---------- 图表 7：延迟分解（堆叠柱） ----------
const chartLatency = computed(() => {
  const rows = data.value?.scale ?? []
  const levels = rows.map((r) => String(r.level))
  return {
    title: {
      text: '查询耗时分解（编码 vs 检索）',
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: levels,
      axisLabel: { color: '#303133' },
    },
    yAxis: {
      type: 'value',
      name: '耗时（ms）',
      axisLabel: { formatter: '{value} ms', color: '#303133' },
      nameTextStyle: { color: '#303133' },
    },
    series: [
      {
        name: '查询编码 embed（ms）',
        type: 'bar',
        stack: 'total',
        barMaxWidth: 50,
        data: rows.map((r) => Number(r.embed_avg_ms.toFixed(1))),
        itemStyle: { color: '#409eff' },
      },
      {
        name: '向量检索 search（ms）',
        type: 'bar',
        stack: 'total',
        barMaxWidth: 50,
        data: rows.map((r) => Number(r.search_avg_ms.toFixed(1))),
        itemStyle: { color: '#79bbff' },
      },
    ],
  }
})

// ---------- 图表 8：结构化查询三方案 ----------
const STRUCT_LABELS: Record<string, string> = {
  sql_exact: '精确过滤',
  keyword_like: '纯关键词',
  vector_only: '纯向量',
}

const chartStructured = computed(() => {
  const rows = data.value?.structured ?? []
  return {
    title: {
      text: '结构化组合条件查询：三方案能力与延迟对照',
      left: 'center',
      textStyle: { fontSize: 14, color: '#303133' },
    },
    tooltip: { trigger: 'axis' },
    legend: { top: 30, textStyle: { color: '#303133' } },
    grid: { left: 10, right: 10, top: 80, bottom: 10, containLabel: true },
    xAxis: {
      type: 'category',
      data: rows.map((r) => `${STRUCT_LABELS[r.mode] ?? r.mode} (n=${r.num_queries})`),
      axisLabel: { color: '#303133' },
    },
    yAxis: {
      type: 'value',
      name: '比例（%）',
      axisLabel: { formatter: '{value}%', color: '#303133' },
      nameTextStyle: { color: '#303133' },
    },
    series: [
      {
        name: '精确率@10（%）',
        type: 'bar',
        barMaxWidth: 36,
        data: rows.map((r) => pct(r['precision@10'])),
        itemStyle: { color: '#409eff' },
      },
      {
        name: '召回率@10（%）',
        type: 'bar',
        barMaxWidth: 36,
        data: rows.map((r) => pct(r['recall@10'])),
        itemStyle: { color: '#79bbff' },
      },
      {
        name: 'F1@10（%）',
        type: 'bar',
        barMaxWidth: 36,
        data: rows.map((r) => pct(r['f1@10'])),
        itemStyle: { color: '#b3d8ff' },
      },
    ],
  }
})
</script>

<style scoped>
.demo-page {
  max-width: 1240px;
  margin: 0 auto;
  color: #303133;
}
.page-head {
  margin-bottom: 12px;
}
.title {
  margin: 0 0 6px;
  font-size: 20px;
  color: #303133;
}
.sub {
  margin: 2px 0;
  font-size: 13px;
  color: #606266;
}
.tabs {
  background: #fff;
  padding: 4px 16px 16px;
  border-radius: 4px;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.stat-card {
  border: 1px solid #e5e7eb;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-bottom: 6px;
}
.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: #409eff;
  line-height: 1.3;
}
.stat-value.small {
  font-size: 15px;
  color: #303133;
}
.chart-card {
  margin-bottom: 12px;
  border: 1px solid #e5e7eb;
}
.source {
  margin: 4px 0 0;
  font-size: 12px;
  color: #909399;
}
.caption {
  margin: 8px 0 0;
  font-size: 12px;
  line-height: 1.7;
  color: #909399;
}
.note-line {
  font-size: 13px;
  line-height: 1.8;
  color: #303133;
}
.mb {
  margin-bottom: 12px;
}
</style>
