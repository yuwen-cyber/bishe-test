import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import SearchCompare from '../views/SearchCompare.vue'
import DemoPanel from '../views/DemoPanel.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: Home },
    { path: '/search', component: SearchCompare },
    { path: '/demo', component: DemoPanel },
  ],
})

export default router
