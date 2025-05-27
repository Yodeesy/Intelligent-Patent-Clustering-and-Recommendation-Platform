<template>
  <n-card title="智能专利推荐" class="max-w-5xl mx-auto mt-6">
    <div class="flex flex-col md:flex-row gap-4 mb-6">
      <n-input
        v-model:value="inputPatentId"
        placeholder="请输入专利ID"
        class="md:w-1/2"
      />
      <n-button type="primary" @click="fetchSimilarPatents">获取推荐</n-button>
      <n-button @click="fetchRandomAndSet">从随机选择一个</n-button>
    </div>

    <n-spin :show="loading">
      <n-grid :x-gap="16" :y-gap="16" :cols="2">
        <n-gi v-for="patent in similarPatents" :key="patent.id">
          <PatentCard :patent="patent" />
        </n-gi>
      </n-grid>
    </n-spin>
  </n-card>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { getRandomPatents, getSimilarPatents } from '@/api/index';
import PatentCard from '@/components/PatentCard.vue';
import { NCard, NInput, NButton, NSpin, NGrid, NGi } from 'naive-ui';

const inputPatentId = ref('');
const similarPatents = ref<any[]>([]);
const loading = ref(false);

const fetchSimilarPatents = async () => {
  if (!inputPatentId.value) return;
  loading.value = true;
  try {
    const response = await getSimilarPatents(inputPatentId.value);
    similarPatents.value = response.data;
  } catch (error) {
    console.error('获取推荐失败:', error);
  } finally {
    loading.value = false;
  }
};

const fetchRandomAndSet = async () => {
  try {
    const response = await getRandomPatents(1);
    if (response.data.length > 0) {
      inputPatentId.value = response.data[0].id;
    }
  } catch (error) {
    console.error('获取随机专利失败:', error);
  }
};
</script>

<style scoped>
</style>