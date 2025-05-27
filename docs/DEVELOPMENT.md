// src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router';
import RecommendView from '@/views/RecommendView.vue';

const routes = [
  {
    path: '/',
    name: 'Recommend',
    component: RecommendView,
  },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
