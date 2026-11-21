<template>
  <div ref="container" class="chart-box" :style="{ height: props.height }"></div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import type { ECharts, EChartsOption } from 'echarts'

const props = withDefaults(
  defineProps<{
    option: Record<string, unknown>
    height?: string
  }>(),
  { height: '340px' },
)

const container = ref<HTMLDivElement | null>(null)
let chart: ECharts | null = null
let observer: ResizeObserver | null = null

function resize() {
  chart?.resize()
}

onMounted(() => {
  if (!container.value) return
  chart = echarts.init(container.value)
  chart.setOption(props.option as EChartsOption, true)
  // 窗口尺寸变化时自适应
  window.addEventListener('resize', resize)
  // 页签切换 / 容器显隐导致的尺寸变化也自适应
  observer = new ResizeObserver(resize)
  observer.observe(container.value)
})

watch(
  () => props.option,
  (option) => {
    chart?.setOption(option as EChartsOption, true)
  },
)

onUnmounted(() => {
  window.removeEventListener('resize', resize)
  observer?.disconnect()
  observer = null
  chart?.dispose()
  chart = null
})
</script>

<style scoped>
.chart-box {
  width: 100%;
}
</style>
