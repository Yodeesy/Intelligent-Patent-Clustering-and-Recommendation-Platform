export const API_CONFIG = {
    SPRING_BOOT_BASE_URL: 'http://localhost:8080/api',
    DJANGO_BASE_URL: 'http://localhost:5000/api',
}

export const API_ENDPOINTS = {
    // Spring Boot endpoints
    AUTH: {
        LOGIN: '/auth/login',
        REGISTER: '/auth/register',
    },
    PATENTS: {
        LIST: '/patents',
        DETAIL: (id: string) => `/patents/${id}`,
        CREATE: '/patents',
        UPDATE: (id: string) => `/patents/${id}`,
        DELETE: (id: string) => `/patents/${id}`,
    },
    
    // Django ML endpoints
    ML: {
        CLUSTER: '/patents/cluster',
        SIMILAR: '/patents/similar',
    }
} 