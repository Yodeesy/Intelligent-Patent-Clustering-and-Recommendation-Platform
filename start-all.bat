@echo off
echo ===== 启动智能专利聚类与推荐平台（Windows） =====

:: 检查 Neo4j
echo 正在检查 Neo4j 数据库是否已在本地启动（bolt://localhost:7687）...
( echo quit ) | neo4j-shell -path bolt://localhost:7687 >nul 2>&1
if %errorlevel% neq 0 (
    echo Neo4j 数据库未启动，请先启动 Neo4j 数据库。
    pause
    exit /b 1
)

:: 检查端口是否可用
set DJANGO_PORT=5000
set VUE_PORT=3000
netstat -ano | findstr ":%DJANGO_PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo 端口 %DJANGO_PORT% 已被占用，请释放该端口后再试。
    pause
    exit /b 1
)
netstat -ano | findstr ":%VUE_PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo 端口 %VUE_PORT% 已被占用，请释放该端口后再试。
    pause
    exit /b 1
)

:: 启动 Spring Boot 后端
echo 启动 Spring Boot 服务...
start cmd /k "cd backend && mvnw spring-boot:run"

:: 启动 Django 服务
echo 启动 Django 服务...
start cmd /k "cd src\\api\\django_app && python manage.py runserver %DJANGO_PORT%"

:: 启动 Vue 前端
echo 启动前端 Vue 应用...
start cmd /k "cd frontend && npm run serve -- --port %VUE_PORT%"

echo 所有服务已尝试启动，请查看新弹出的终端窗口。
pause