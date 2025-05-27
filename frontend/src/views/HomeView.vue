<template>
  <div class="relative min-h-screen bg-gradient-to-tr from-sky-900 via-indigo-900 to-gray-900 flex items-center justify-center">
    <!-- 欢迎卡片 -->
    <transition name="fade-zoom">
      <n-card
        v-if="!leaving"
        class="relative z-10 max-w-3xl w-full mx-auto shadow-xl bg-white/90 rounded-2xl"
        hoverable
      >
        <div class="text-center px-6 py-10">
          <h2 class="text-3xl font-bold mb-4 text-gray-900">欢迎使用智能专利推荐平台</h2>
          <p class="text-gray-700 text-lg mb-8">
            本平台融合了聚类分析与相似度推荐，助力专利数据的智能化挖掘与应用。
          </p>
          <n-button
            type="primary"
            size="large"
            @click="startTransition"
            class="transition-all duration-300 hover:scale-105"
          >
            Enter
          </n-button>
        </div>
      </n-card>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { NCard, NButton } from 'naive-ui'

const router = useRouter()
const leaving = ref(false)

const startTransition = () => {
  leaving.value = true
  setTimeout(() => {
    router.push('/recommend')
  }, 500)
}
</script>

<style scoped>
.min-h-screen {
  min-height: 100vh;
}

/* 动画过渡 */
.fade-zoom-enter-active,
.fade-zoom-leave-active {
  transition: all 0.5s ease;
}
.fade-zoom-enter-from,
.fade-zoom-leave-to {
  opacity: 0;
  transform: scale(0.9);
}
.fade-zoom-enter-to,
.fade-zoom-leave-from {
  opacity: 1;
  transform: scale(1);
}
</style>
