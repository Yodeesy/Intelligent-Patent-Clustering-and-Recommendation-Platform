import axios, { AxiosRequestConfig } from 'axios';
import { API_CONFIG, API_ENDPOINTS } from './config';

// Spring Boot API client
const springBootApi = axios.create({
    baseURL: API_CONFIG.SPRING_BOOT_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Django API client
const djangoApi = axios.create({
    baseURL: API_CONFIG.DJANGO_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add token to requests
springBootApi.interceptors.request.use((config: AxiosRequestConfig) => {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
        config.headers = {
            ...config.headers,
            Authorization: `Bearer ${token}`
        };
    }
    return config;
});

export const authApi = {
    login: (username: string, password: string) =>
        springBootApi.post(API_ENDPOINTS.AUTH.LOGIN, { username, password }),
    register: (username: string, password: string) =>
        springBootApi.post(API_ENDPOINTS.AUTH.REGISTER, { username, password }),
};

export const patentApi = {
    // Spring Boot endpoints
    getPatents: () => springBootApi.get(API_ENDPOINTS.PATENTS.LIST),
    getPatent: (id: string) => springBootApi.get(API_ENDPOINTS.PATENTS.DETAIL(id)),
    createPatent: (data: any) => springBootApi.post(API_ENDPOINTS.PATENTS.CREATE, data),
    updatePatent: (id: string, data: any) => springBootApi.put(API_ENDPOINTS.PATENTS.UPDATE(id), data),
    deletePatent: (id: string) => springBootApi.delete(API_ENDPOINTS.PATENTS.DELETE(id)),
    
    // Django ML endpoints
    clusterPatents: (patents: any[], n_clusters: number = 5) =>
        djangoApi.post(API_ENDPOINTS.ML.CLUSTER, { patents, n_clusters }),
    getSimilarPatents: (patentId: string, top_k: number = 10) =>
        djangoApi.get(API_ENDPOINTS.ML.SIMILAR, { params: { patent_id: patentId, top_k } }),
}; 