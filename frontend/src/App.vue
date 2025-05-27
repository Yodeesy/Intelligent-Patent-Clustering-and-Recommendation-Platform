<!-- src/App.vue -->
<template>
  <div id="app">
    <header>
      <h1>智能专利推荐平台</h1>
      <nav>
        <router-link to="/">推荐</router-link>
        <router-link to="/cluster">聚类</router-link>
      </nav>
    </header>
    <main>
      <router-view />
    </main>
    <footer>
      <p>© 2025 Intelligent Patent Platform</p>
    </footer>
  </div>
</template>

<script lang="ts" setup>
</script>

<style scoped>
#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

header {
  background-color: #2c3e50;
  color: white;
  padding: 1rem;
  text-align: center;
}

nav {
  margin-top: 0.5rem;
}
nav a {
  color: #fff;
  margin: 0 1rem;
  text-decoration: none;
}
nav a.router-link-exact-active {
  font-weight: bold;
  text-decoration: underline;
}

main {
  flex: 1;
  padding: 2rem;
}

footer {
  background-color: #f2f2f2;
  text-align: center;
  padding: 0.5rem;
  font-size: 0.9rem;
}
</style>