<template>
  <el-config-provider>
    <div class="app-container">
      <el-container>
        <el-header>
          <h1>专利聚类与推荐系统</h1>
        </el-header>
        <el-main>
          <router-view></router-view>
        </el-main>
      </el-container>
    </div>
  </el-config-provider>
</template>

<script setup>
import { ElConfigProvider } from 'element-plus'
</script>

<style lang="scss">
.app-container {
  min-height: 100vh;
  .el-header {
    background-color: #409EFF;
    color: white;
    line-height: 60px;
    text-align: center;
  }
  .el-main {
    padding: 20px;
    background-color: #f5f7fa;
  }
}
</style> 