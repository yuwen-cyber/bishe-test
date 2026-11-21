<template>
  <div class="search-page">
    <el-card shadow="never" class="filter-card">
      <div class="filters">
        <el-input
          v-model="query"
          placeholder="输入检索语句…"
          clearable
          class="query-input"
          @keyup.enter="doSearch"
        />
        <el-select
          v-model="selectedEmotions"
          multiple
          collapse-tags
          collapse-tags-tooltip
          placeholder="情绪标签"
          class="emotion-select"
        >
          <el-option v-for="e in emotions" :key="e" :label="e" :value="e" />
        </el-select>
        <el-date-picker
          v-model="timeRange"
          type="datetimerange"
          start-placeholder="开始时间"
          end-placeholder="结束时间"
          class="time-picker"
        />
        <el-button type="primary" :loading="loading" @click="doSearch">搜索对比</el-button>
      </div>
    </el-card>

    <div class="columns">
      <div class="column">
        <div class="col-header">
          <h3>基准方案 · MySQL 全文检索</h3>
          <el-tag v-if="baseline" type="info">{{ baseline.retrieval_time_ms.toFixed(2) }} ms</el-tag>
        </div>
        <div class="col-body">
          <ResultCard v-for="item in baseline?.results || []" :key="item.segment_id" :item="item" />
          <el-empty v-if="baseline && baseline.results.length === 0" description="无结果" />
          <el-empty v-if="!baseline" description="点击“搜索对比”开始检索" :image-size="80" />
        </div>
      </div>

      <div class="column">
        <div class="col-header">
          <h3>高性能方案 · Milvus 向量检索</h3>
          <el-tag v-if="advanced" type="success">{{ advanced.retrieval_time_ms.toFixed(2) }} ms</el-tag>
        </div>
        <div class="col-body">
          <ResultCard v-for="item in advanced?.results || []" :key="'a' + item.segment_id" :item="item" />
          <el-empty v-if="advanced && advanced.results.length === 0" description="无结果" />
          <el-empty v-if="!advanced" description="点击“搜索对比”开始检索" :image-size="80" />
        </div>
      </div>

      <div class="column">
        <div class="col-header">
          <h3>混合方案 · SQL+向量 RRF 融合</h3>
          <el-tag v-if="hybrid" type="warning">{{ hybrid.retrieval_time_ms.toFixed(2) }} ms</el-tag>
        </div>
        <div class="col-body">
          <ResultCard v-for="item in hybrid?.results || []" :key="'h' + item.segment_id" :item="item" />
          <el-empty v-if="hybrid && hybrid.results.length === 0" description="无结果" />
          <el-empty v-if="!hybrid" description="点击“搜索对比”开始检索" :image-size="80" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import ResultCard from '../components/ResultCard.vue'
import { advancedSearch, baselineSearch, hybridSearch, type SearchResponse } from '../api'

const emotions = ['喜悦', '愤怒', '悲伤', '恐惧', '惊讶', '厌恶', '中性']

const query = ref('')
const selectedEmotions = ref<string[]>([])
const timeRange = ref<[Date, Date] | null>(null)
const loading = ref(false)
const baseline = ref<SearchResponse | null>(null)
const advanced = ref<SearchResponse | null>(null)
const hybrid = ref<SearchResponse | null>(null)

function toLocalIso(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

async function doSearch() {
  if (!query.value.trim()) {
    ElMessage.warning('请输入检索语句')
    return
  }
  const params: Record<string, string> = { q: query.value.trim() }
  if (selectedEmotions.value.length) params.emotions = selectedEmotions.value.join(',')
  if (timeRange.value && timeRange.value[0] && timeRange.value[1]) {
    params.start = toLocalIso(timeRange.value[0])
    params.end = toLocalIso(timeRange.value[1])
  }

  loading.value = true
  try {
    const [b, a, h] = await Promise.all([baselineSearch(params), advancedSearch(params), hybridSearch(params)])
    baseline.value = b
    advanced.value = a
    hybrid.value = h
  } catch (e: any) {
    ElMessage.error('检索失败：' + (e?.response?.data?.detail || e?.message || '未知错误'))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.filter-card {
  margin-bottom: 16px;
}
.filters {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.query-input {
  flex: 1;
  min-width: 240px;
}
.emotion-select {
  width: 220px;
}
.time-picker {
  width: 360px;
}
.columns {
  display: flex;
  gap: 16px;
  height: calc(100vh - 190px);
}
.column {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
}
.col-header {
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.col-header h3 {
  margin: 0;
  font-size: 15px;
  color: #303133;
}
.col-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
</style>