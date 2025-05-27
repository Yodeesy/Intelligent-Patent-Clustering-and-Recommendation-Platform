<template>
  <n-card title="专利聚类结果" class="max-w-5xl mx-auto mt-6">
    <n-spin :show="loading">
      <n-alert v-if="error" type="error" class="mb-4">{{ error }}</n-alert>
      <n-collapse v-if="!error">
        <n-collapse-item
          v-for="(patents, clusterId) in clusterResult"
          :key="clusterId"
          :title="`聚类组 ${clusterId}`"
        >
          <n-grid :x-gap="12" :y-gap="12" :cols="2">
            <n-gi v-for="patent in patents" :key="patent.id || patent.patent_id">
              <PatentCard :patent="patent" />
            </n-gi>
          </n-grid>
        </n-collapse-item>
      </n-collapse>
    </n-spin>
  </n-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { clusterPatents } from '@/api/index';
import PatentCard from '@/components/PatentCard.vue';
import {
  NCard,
  NSpin,
  NAlert,
  NGrid,
  NGi,
  NCollapse,
  NCollapseItem
} from 'naive-ui';

const clusterResult = ref<Record<string, any[]>>({});
const loading = ref(true);
const error = ref<string | null>(null);

const examplePatentIds = ['US123456A', 'CN654321B', 'JP789012C', 'US888999D'];

onMounted(async () => {
  try {
    const response = await clusterPatents(examplePatentIds);
    clusterResult.value = response;
  } catch (err) {
    error.value = '获取聚类结果失败。';
    console.error(err);
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
</style>
