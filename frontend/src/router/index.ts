import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import PatentList from '../views/PatentList.vue'
import ClusterResult from '../views/ClusterResult.vue'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'PatentList',
    component: PatentList
  },
  {
    path: '/cluster',
    name: 'ClusterResult',
    component: ClusterResult
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router 