<template>
  <el-card class="home-card" shadow="never">
    <template #header>
      <span>对话录制</span>
    </template>

    <div class="video-wrap">
      <video ref="previewEl" autoplay playsinline muted></video>
      <div v-if="!recording" class="video-placeholder">点击“开始对话”开启摄像头与麦克风</div>
    </div>

    <div class="actions">
      <el-button type="primary" :loading="uploading" :disabled="recording" @click="start">
        开始对话
      </el-button>
      <el-button type="danger" :disabled="!recording" @click="stop">
        结束对话
      </el-button>
    </div>

    <el-alert v-if="saved" title="对话已保存" type="success" show-icon :closable="false" class="tip" />
    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" class="tip" />
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { uploadConversation } from '../api'

const previewEl = ref<HTMLVideoElement>()
const recording = ref(false)
const uploading = ref(false)
const saved = ref(false)
const error = ref('')

let videoStream: MediaStream | null = null
let audioStream: MediaStream | null = null
let videoRecorder: MediaRecorder | null = null
let audioRecorder: MediaRecorder | null = null
let videoChunks: BlobPart[] = []
let audioChunks: BlobPart[] = []
let startTime = 0
let stopped = 0

function pickMime(kind: 'video' | 'audio'): string | undefined {
  const opts =
    kind === 'video'
      ? ['video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4']
      : ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
  return opts.find((m) => MediaRecorder.isTypeSupported(m))
}

async function start() {
  error.value = ''
  saved.value = false
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch {
    error.value = '无法访问摄像头或麦克风，请在浏览器中授权后重试。'
    return
  }

  if (previewEl.value) previewEl.value.srcObject = videoStream

  videoChunks = []
  audioChunks = []
  stopped = 0

  videoRecorder = new MediaRecorder(videoStream, { mimeType: pickMime('video') })
  audioRecorder = new MediaRecorder(audioStream, { mimeType: pickMime('audio') })

  videoRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) videoChunks.push(e.data)
  }
  audioRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data)
  }

  const onStop = () => {
    stopped += 1
    if (stopped >= 2) handleFinish()
  }
  videoRecorder.onstop = onStop
  audioRecorder.onstop = onStop

  videoRecorder.start()
  audioRecorder.start()
  recording.value = true
  startTime = Date.now()
}

function stop() {
  if (videoRecorder && videoRecorder.state !== 'inactive') videoRecorder.stop()
  if (audioRecorder && audioRecorder.state !== 'inactive') audioRecorder.stop()
}

function stopStreams() {
  videoStream?.getTracks().forEach((t) => t.stop())
  audioStream?.getTracks().forEach((t) => t.stop())
  videoStream = null
  audioStream = null
  if (previewEl.value) previewEl.value.srcObject = null
}

async function handleFinish() {
  stopStreams()
  recording.value = false
  const durationMs = Date.now() - startTime

  const videoBlob = new Blob(videoChunks, { type: pickMime('video') || 'video/webm' })
  const audioBlob = new Blob(audioChunks, { type: pickMime('audio') || 'audio/webm' })
  if (videoBlob.size === 0 || audioBlob.size === 0) {
    error.value = '未能录制到内容，请重试。'
    return
  }

  const form = new FormData()
  form.append('video', videoBlob, 'video.webm')
  form.append('audio', audioBlob, 'audio.webm')
  form.append('duration_ms', String(durationMs))

  uploading.value = true
  try {
    const res = await uploadConversation(form)
    saved.value = true
    ElMessage.success(res.message)
  } catch (e: any) {
    error.value = '上传失败：' + (e?.response?.data?.detail || e?.message || '未知错误')
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.home-card {
  max-width: 720px;
  margin: 0 auto;
}
.video-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}
.video-wrap video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.video-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
}
.actions {
  margin-top: 16px;
  display: flex;
  gap: 12px;
}
.tip {
  margin-top: 16px;
}
</style>