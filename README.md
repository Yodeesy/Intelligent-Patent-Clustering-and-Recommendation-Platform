# Intelligent Patent Clustering and Recommendation Platform

## 项目简介
这是一个基于图神经网络和机器学习的智能专利聚类与推荐平台。该平台采用微服务架构，将业务逻辑和机器学习服务分离，提供专利文献的智能分析、聚类和推荐功能。

## 主要功能
- 专利聚类分析：使用图神经网络(RGCN)和聚类算法对专利进行智能分组
- 相似专利推荐：基于专利内容和关键词的相似度分析，推荐相关专利
- 可视化展示：直观展示专利聚类结果和关系网络
- 随机专利浏览：支持随机获取专利列表，便于探索发现

## 技术架构
本项目采用微服务架构，包含以下主要组件：

### Spring Boot服务 (8080端口)
- 用户认证和授权管理
- 专利基本信息的CRUD操作
- Neo4j图数据库交互
- RESTful API接口

### Django服务 (5000端口)
- 机器学习模型训练和预测
- 专利聚类分析服务
- 相似专利推荐服务
- ML API接口

### 前端 (3000端口)
- Vue.js前端框架
- Element Plus UI组件库
- ECharts数据可视化
- 与Spring Boot和Django服务交互

### 数据存储
- Neo4j (7687端口)：存储专利关系网络
- SQLite：Django默认数据库，存储ML相关数据

### 技术栈详情
- Spring Boot 3.4.5
- Django 5.2
- Vue.js 3.x
- Neo4j 4.x
- PyTorch + scikit-learn

## 系统要求
- JDK 21+
- Python 3.8+ (推荐使用Anaconda/Miniconda)
- Neo4j 4.x
- Node.js 14+
- npm 6+

## 项目结构
```
.
├── backend/                # Spring Boot后端项目
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/     # Java源代码
│   │   │   └── resources/ # 配置文件
│   │   └── test/         # 测试代码
├── frontend/              # Vue.js前端项目
│   ├── src/
│   │   ├── components/   # 通用组件
│   │   ├── views/        # 页面组件
│   │   ├── store/        # Vuex状态管理
│   │   └── router/       # 路由配置
├── src/                   # Python服务
│   ├── api/              # Django API服务
│   │   ├── django_app/   # Django项目目录
│   │   ├── requirements.txt # Python依赖
│   │   └── manage.py     # Django管理脚本
│   ├── models/           # 机器学习模型
│   └── utils/            # 工具函数
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── scripts/              # 工具脚本
├── monitoring/           # 监控配置
│   ├── prometheus.yml    # Prometheus配置
│   └── logstash.conf    # Logstash配置
├── docs/                 # 项目文档
│   ├── DEVELOPMENT.md   # 开发指南
│   └── DEPLOYMENT.md    # 部署文档
├── docker-compose.yml    # Docker编排配置
├── environment.yml       # Python环境配置
├── start-all.bat         # 适用于 Windows 快速启动
├── start-all.sh          # 适用于 Linux/macOS 快速启动
└── README.md            # 项目说明
```

## 安装和启动
1. 克隆项目
```bash
git clone https://github.com/yourusername/Intelligent-Patent-Clustering-and-Recommendation-Platform.git
cd Intelligent-Patent-Clustering-and-Recommendation-Platform
```

2. 安装依赖

### 方式一：使用Conda环境（推荐）
```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate patent-platform

# 安装Spring Boot依赖
cd backend
# Windows系统使用:
mvnw install
# Linux/Mac系统使用:
./mvnw install

# 安装Python依赖
cd ../src/api
pip install -r requirements.txt

# 安装前端依赖
cd ../../frontend
npm install
```

3. 启动服务
### 快速启动（推荐）

请先确保 Neo4j 数据库已启动，再运行以下命令之一：

#### Windows 用户
```bash
start-all.bat
```

#### Linux / macOS 用户
```shell
./start-all.sh
```

### 手动启动服务（以下为Windows用户示例）
#### 第一步：启动Neo4j数据库
确保Neo4j数据库已启动并运行在默认端口(7687)

#### 第二步：启动Spring Boot服务 (8080端口)
```bash
cd backend
# Windows系统使用:
mvnw spring-boot:run
# Linux/Mac系统使用:
./mvnw spring-boot:run
```

#### 第三步：启动Django服务 (5000端口)
```bash
cd src/api/django_app
python manage.py migrate  # 首次运行需要执行数据库迁移
python manage.py runserver 5000
```

#### 第四步：启动前端服务 (3000端口)
```bash
cd frontend
# 修改 package.json 中的 serve 脚本，添加端口配置
npm run serve -- --port 3000
```

## 服务地址
- 前端界面：`http://localhost:3000`
- Spring Boot API：`http://localhost:8080`
  - Swagger文档：`http://localhost:8080/swagger-ui.html`
- Django API：`http://localhost:5000`
  - Swagger文档：`http://localhost:5000/swagger/`
- Neo4j数据库：`bolt://localhost:7687`

## 开发指南
请参考 [开发指南](docs/DEVELOPMENT.md)

## 部署指南
请参考 [部署指南](docs/DEPLOYMENT.md)

## 贡献指南
欢迎提交Issue和Pull Request。在提交PR之前，请确保：
1. 代码符合项目的编码规范
2. 添加必要的测试用例
3. 更新相关文档

## 许可证
本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。
