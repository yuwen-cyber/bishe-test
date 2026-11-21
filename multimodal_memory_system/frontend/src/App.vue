<template>
  <el-container class="layout">
    <el-header class="header" height="60px">
      <div class="brand">多模态记忆系统</div>
      <el-menu mode="horizontal" :default-active="$route.path" router :ellipsis="false" class="menu">
        <el-menu-item index="/">对话录制</el-menu-item>
        <el-menu-item index="/search">检索对比</el-menu-item>
        <el-menu-item index="/demo">演示面板</el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup lang="ts"></script>

<style>
body {
  margin: 0;
  font-family: system-ui, -apple-system, 'Segoe UI', 'PingFang SC', sans-serif;
  background: #f5f7fa;
}
.layout {
  height: 100vh;
}
.header {
  background: #fff;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 0 24px;
}
.brand {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}
.menu {
  border-bottom: none;
  flex: 1;
  min-width: 0;
}
.main {
  overflow: auto;
  padding: 20px;
}
</style>