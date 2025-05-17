<template>
  <div class="analysis-results">
    <el-card>
      <template #header>
        <div class="card-header">
          <h2>分析结果</h2>
          <el-button type="primary" @click="$router.push('/')">返回列表</el-button>
        </div>
      </template>

      <div v-if="results" class="results-content">
        <!-- 聚类可视化 -->
        <div class="visualization-container">
          <div ref="chartRef" style="width: 100%; height: 400px;"></div>
        </div>

        <!-- 聚类结果列表 -->
        <div class="clusters-list">
          <el-collapse v-model="activeNames">
            <el-collapse-item
              v-for="(cluster, index) in groupedResults"
              :key="index"
              :title="'聚类 ' + (index + 1) + ' (' + cluster.length + ' 个专利)'"
              :name="index">
              <div v-for="result in cluster" :key="result.patent_id" class="patent-item">
                <h4>{{ result.patent_id }}</h4>
                <div class="similar-patents">
                  <h5>相似专利：</h5>
                  <el-table :data="result.similar_patents" style="width: 100%">
                    <el-table-column prop="pubno" label="专利号" width="180" />
                    <el-table-column prop="title" label="标题" />
                    <el-table-column prop="similarity" label="相似度" width="100">
                      <template #default="{ row }">
                        {{ (row.similarity * 100).toFixed(2) }}%
                      </template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
      <el-empty v-else description="暂无分析结果" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import * as echarts from 'echarts'

const results = ref(null)
const activeNames = ref([0])
const chartRef = ref(null)
let chart = null

// 从localStorage获取分析结果
onMounted(() => {
  const savedResults = localStorage.getItem('analysisResults')
  if (savedResults) {
    results.value = JSON.parse(savedResults)
    initChart()
  }
})

// 按聚类分组结果
const groupedResults = computed(() => {
  if (!results.value?.results) return []
  
  const groups = {}
  results.value.results.forEach(result => {
    if (!groups[result.cluster_id]) {
      groups[result.cluster_id] = []
    }
    groups[result.cluster_id].push(result)
  })
  
  return Object.values(groups)
})

// 初始化图表
const initChart = () => {
  if (!chartRef.value || !results.value) return

  const chartData = processDataForChart(results.value.results)
  chart = echarts.init(chartRef.value)
  
  const option = {
    title: {
      text: '专利聚类可视化',
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      formatter: '{b}'
    },
    series: [{
      type: 'graph',
      layout: 'force',
      data: chartData.nodes,
      links: chartData.links,
      categories: chartData.categories,
      roam: true,
      label: {
        show: true,
        position: 'right'
      },
      force: {
        repulsion: 100
      }
    }]
  }
  
  chart.setOption(option)
}

// 处理图表数据
const processDataForChart = (results) => {
  const nodes = []
  const links = []
  const categories = []
  
  // 创建聚类类别
  const uniqueClusters = [...new Set(results.map(r => r.cluster_id))]
  uniqueClusters.forEach(clusterId => {
    categories.push({ name: `聚类 ${clusterId + 1}` })
  })
  
  // 创建节点和连接
  results.forEach(result => {
    // 添加主专利节点
    nodes.push({
      name: result.patent_id,
      category: result.cluster_id,
      value: 20
    })
    
    // 添加相似专利节点和连接
    result.similar_patents.forEach(similar => {
      nodes.push({
        name: similar.pubno,
        category: result.cluster_id,
        value: similar.similarity * 15
      })
      
      links.push({
        source: result.patent_id,
        target: similar.pubno,
        value: similar.similarity
      })
    })
  })
  
  return { nodes, links, categories }
}
</script>

<style lang="scss" scoped>
.analysis-results {
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .results-content {
    .visualization-container {
      margin-bottom: 20px;
      border: 1px solid #ebeef5;
      border-radius: 4px;
    }
    
    .patent-item {
      margin-bottom: 20px;
      padding: 10px;
      border: 1px solid #ebeef5;
      border-radius: 4px;
      
      h4 {
        margin: 0 0 10px;
      }
      
      .similar-patents {
        h5 {
          margin: 10px 0;
        }
      }
    }
  }
}
</style> 