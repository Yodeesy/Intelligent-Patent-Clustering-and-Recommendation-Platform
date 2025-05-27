<!-- src/components/PatentCard.vue -->
<template>
  <n-card
    :title="patent.title || '未命名专利'"
    size="medium"
    hoverable
    class="patent-card"
  >
    <n-thing>
      <n-grid :cols="1" :x-gap="8" :y-gap="4">
        <n-gi>
          <strong>专利号：</strong>{{ patent.id || patent.patent_id || '未知' }}
        </n-gi>
        <n-gi>
          <strong>摘要：</strong>{{ patent.abstract || patent.abstractText || '暂无摘要' }}
        </n-gi>
        <n-gi>
          <strong>发明人：</strong>
          <span>{{ Array.isArray(patent.inventors) ? patent.inventors.join(', ') : '未知' }}</span>
        </n-gi>
        <n-gi>
          <strong>申请人：</strong>
          <span>{{ Array.isArray(patent.applicants) ? patent.applicants.join(', ') : '未知' }}</span>
        </n-gi>
        <n-gi>
          <strong>公开日：</strong>{{ patent.publicationDate || patent.publication_date || '未知' }}
        </n-gi>
      </n-grid>
    </n-thing>
  </n-card>
</template>

<script lang="ts" setup>
import { NCard, NThing, NGrid, NGi } from 'naive-ui'

interface Patent {
  id?: string
  patent_id?: string
  title?: string
  abstract?: string
  abstractText?: string
  inventors?: string[]
  applicants?: string[]
  publicationDate?: string
  publication_date?: string
}

defineProps<{ patent: Patent }>()
</script>

<style scoped>
.patent-card {
  margin-bottom: 1.5rem;
  transition: transform 0.2s ease;
}
.patent-card:hover {
  transform: translateY(-4px);
}
</style>
