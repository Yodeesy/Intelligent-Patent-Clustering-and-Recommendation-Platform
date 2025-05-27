import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL, // 动态配置
  timeout: 10000,
})

export const getRandomPatents = (size: number = 100) => {
  return api.get('/random', { params: { size } })
}

export const getSimilarPatents = (patentId: string, limit: number = 10) => {
  return api.get(`/${patentId}/similar`, { params: { limit } })
}

export const clusterPatents = (patentIds: string[]) => {
  return api.post('/cluster', patentIds)
}
