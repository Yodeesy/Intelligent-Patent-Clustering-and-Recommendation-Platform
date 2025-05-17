import { createRouter, createWebHistory } from 'vue-router'
import PatentList from '../views/PatentList.vue'
import AnalysisResults from '../views/AnalysisResults.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: PatentList
    },
    {
      path: '/results',
      name: 'results',
      component: AnalysisResults
    }
  ]
})

export default router 