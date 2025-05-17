<template>
  <div class="cluster-result">
    <h1>专利聚类结果</h1>
    <div v-if="loading" class="loading">
      加载中...
    </div>
    <div v-else class="cluster-container">
      <!-- 这里将添加聚类结果的可视化 -->
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import { useStore } from 'vuex'

export default defineComponent({
  name: 'ClusterResult',
  setup() {
    const store = useStore()
    return {
      loading: false
    }
  }
})
</script>

<style scoped>
.cluster-result {
  padding: 20px;
}

.loading {
  text-align: center;
  padding: 20px;
}

.cluster-container {
  margin-top: 20px;
}
</style> 