<template>
  <div class="patent-list">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>专利列表</span>
          <el-button
            type="primary"
            :disabled="!selectedPatents.length"
            @click="handleCluster"
          >
            开始聚类分析
          </el-button>
        </div>
      </template>
      
      <el-table
        v-loading="loading"
        :data="patents"
        style="width: 100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="pubNo" label="公开号" width="180" />
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="pubTime" label="公开日期" width="180" />
        <el-table-column prop="countryName" label="国家/地区" width="120" />
        <el-table-column fixed="right" label="操作" width="120">
          <template #default="{ row }">
            <el-button
              link
              type="primary"
              @click="handleViewDetails(row)"
            >
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 专利详情对话框 -->
    <el-dialog
      v-model="dialogVisible"
      title="专利详情"
      width="70%"
    >
      <template v-if="currentPatent">
        <h3>{{ currentPatent.title }}</h3>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="公开号">
            {{ currentPatent.pubNo }}
          </el-descriptions-item>
          <el-descriptions-item label="公开日期">
            {{ currentPatent.pubTime }}
          </el-descriptions-item>
          <el-descriptions-item label="国家/地区">
            {{ currentPatent.countryName }}
          </el-descriptions-item>
          <el-descriptions-item label="来源数据库">
            {{ currentPatent.srcDatabase }}
          </el-descriptions-item>
          <el-descriptions-item label="摘要" :span="2">
            {{ currentPatent.summary }}
          </el-descriptions-item>
          <el-descriptions-item label="主权项" :span="2">
            {{ currentPatent.claims }}
          </el-descriptions-item>
        </el-descriptions>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, ref, computed } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { Patent } from '@/store'
import { ElMessage } from 'element-plus'

export default defineComponent({
  name: 'PatentList',
  
  setup() {
    const store = useStore()
    const router = useRouter()
    const dialogVisible = ref(false)
    const currentPatent = ref<Patent | null>(null)
    const selectedPatents = ref<string[]>([])

    const handleSelectionChange = (selection: Patent[]) => {
      selectedPatents.value = selection.map(item => item.pubNo)
      store.commit('setSelectedPatents', selectedPatents.value)
    }

    const handleViewDetails = (patent: Patent) => {
      currentPatent.value = patent
      dialogVisible.value = true
    }

    const handleCluster = async () => {
      if (selectedPatents.value.length < 2) {
        ElMessage.warning('请至少选择两个专利进行聚类分析')
        return
      }
      await store.dispatch('clusterPatents')
      router.push('/cluster')
    }

    onMounted(() => {
      store.dispatch('fetchRandomPatents')
    })

    return {
      patents: computed(() => store.state.patents),
      loading: computed(() => store.state.loading),
      selectedPatents,
      dialogVisible,
      currentPatent,
      handleSelectionChange,
      handleViewDetails,
      handleCluster
    }
  }
})
</script>

<style scoped>
.patent-list {
  max-width: 1200px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style> 