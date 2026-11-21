<template>
  <el-card shadow="hover" class="result-card">
    <div class="meta">
      <el-tag :type="emotionType(item.emotion)" size="small">{{ item.emotion || '未知' }}</el-tag>
      <el-tag v-if="item.data_source === 'noise'" type="info" size="small" effect="plain">
        噪声
      </el-tag>
      <el-tag v-else-if="item.data_source === 'user'" type="primary" size="small" effect="plain">
        自录
      </el-tag>
      <span class="time">{{ fmtTime(item.timestamp) }}</span>
    </div>
    <div class="text">{{ item.asr_text }}</div>
    <div class="media">
      <audio v-if="item.audio_url" :src="item.audio_url" controls preload="metadata"></audio>
      <video
        v-if="item.video_url"
        ref="videoEl"
        :src="item.video_url"
        controls
        preload="metadata"
        @play="onVideoPlay"
        @pause="onVideoPause"
        @seeked="onVideoSeek"
        @volumechange="onVideoVolume"
      ></video>
      <!-- 隐藏音轨：仅用于给视频配音，视频播放时联动 -->
      <audio
        v-if="item.video_url && item.audio_url"
        ref="audioEl"
        :src="item.audio_url"
        preload="metadata"
        class="hidden-audio"
      ></audio>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { SearchResultItem } from '../api'

const props = defineProps<{ item: SearchResultItem }>()

const videoEl = ref<HTMLVideoElement>()
const audioEl = ref<HTMLAudioElement>()

// 视频本身不含音轨（录制时音视频分开），播放视频时联动同步播放音频，实现“视频带声”
function onVideoPlay() {
  const v = videoEl.value
  const a = audioEl.value
  if (!v || !a) return
  a.currentTime = v.currentTime
  a.play().catch(() => {})
}
function onVideoPause() {
  audioEl.value?.pause()
}
function onVideoSeek() {
  const v = videoEl.value
  const a = audioEl.value
  if (!v || !a) return
  a.currentTime = v.currentTime
}
function onVideoVolume() {
  const v = videoEl.value
  const a = audioEl.value
  if (!v || !a) return
  a.volume = v.volume
  a.muted = v.muted
}

function emotionType(emotion: string | null): 'success' | 'danger' | 'warning' | 'info' | 'primary' {
  switch (emotion) {
    case '喜悦':
      return 'success'
    case '愤怒':
    case '悲伤':
    case '恐惧':
      return 'danger'
    case '惊讶':
      return 'warning'
    default:
      return 'info'
  }
}

function fmtTime(ts: string | null): string {
  if (!ts) return ''
  const d = new Date(ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}
</script>

<style scoped>
.result-card {
  margin-bottom: 12px;
}
.meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.time {
  color: #909399;
  font-size: 13px;
}
.text {
  color: #303133;
  line-height: 1.6;
  margin-bottom: 12px;
}
.media {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.hidden-audio {
  display: none;
}
.media video {
  width: 100%;
  max-height: 240px;
  background: #000;
  border-radius: 6px;
}
</style>