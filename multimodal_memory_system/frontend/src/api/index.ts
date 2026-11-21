import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

export interface SearchResultItem {
  segment_id: number
  conversation_id: number
  timestamp: string | null
  emotion: string | null
  asr_text: string
  audio_url: string | null
  video_url: string | null
  score: number | null
  // 数据来源：formal(正式) / noise(噪声) / user(自录) / unknown
  data_source?: 'formal' | 'noise' | 'user' | 'unknown'
}

export interface SearchResponse {
  retrieval_time_ms: number
  total: number
  results: SearchResultItem[]
}

export interface UploadResponse {
  conversation_id: number
  segment_id: number
  emotion: string | null
  asr_text: string | null
  message: string
}

export async function uploadConversation(form: FormData): Promise<UploadResponse> {
  const { data } = await http.post<UploadResponse>('/upload', form)
  return data
}

export async function baselineSearch(params: Record<string, string>): Promise<SearchResponse> {
  const { data } = await http.get<SearchResponse>('/baseline/search', { params })
  return data
}

export async function advancedSearch(params: Record<string, string>): Promise<SearchResponse> {
  const { data } = await http.get<SearchResponse>('/advanced/search', { params })
  return data
}

export async function hybridSearch(params: Record<string, string>): Promise<SearchResponse> {
  const { data } = await http.get<SearchResponse>('/hybrid/search', { params })
  return data
}