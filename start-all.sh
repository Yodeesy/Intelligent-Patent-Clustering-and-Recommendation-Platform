#!/bin/bash
echo "===== 启动智能专利聚类与推荐平台（Linux/macOS） ====="

# 检查 Neo4j
echo "正在检查 Neo4j 数据库是否已在本地启动（bolt://localhost:7687）..."
nc -z localhost 7687
if [ $? -ne 0 ]; then
    echo "Neo4j 数据库未启动，请先启动 Neo4j 数据库。"
    exit 1
fi

# 检查端口是否可用
DJANGO_PORT=5000
VUE_PORT=3000
if lsof -Pi :$DJANGO_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "端口 $DJANGO_PORT 已被占用，请释放该端口后再试。"
    exit 1
fi
if lsof -Pi :$VUE_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "端口 $VUE_PORT 已被占用，请释放该端口后再试。"
    exit 1
fi

# 启动 Spring Boot
echo "启动 Spring Boot 服务..."
(cd backend && ./mvnw spring-boot:run) &
SPRING_PID=$!

# 启动 Django
echo "启动 Django 服务..."
(cd src/api/django_app && python manage.py runserver $DJANGO_PORT) &
DJANGO_PID=$!

# 启动 Vue
echo "启动前端 Vue 应用..."
(cd frontend && npm run serve -- --port $VUE_PORT) &
VUE_PID=$!

# 记录进程 ID 到文件，方便后续管理
echo "$SPRING_PID $DJANGO_PID $VUE_PID" > service_pids.txt

echo "所有服务已后台启动，请查看各终端输出日志。"