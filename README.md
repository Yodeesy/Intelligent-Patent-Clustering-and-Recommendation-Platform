# Intelligent Patent Clustering and Recommendation Platform

## 项目简介
这是一个基于图神经网络和机器学习的智能专利聚类与推荐平台。该平台能够对专利文献进行智能分析、聚类和推荐，帮助用户更好地理解和利用专利信息。

## 主要功能
- 专利聚类分析：使用图神经网络(RGCN)和聚类算法对专利进行智能分组
- 相似专利推荐：基于专利内容和关键词的相似度分析，推荐相关专利
- 可视化展示：直观展示专利聚类结果和关系网络
- 随机专利浏览：支持随机获取专利列表，便于探索发现

## 技术架构
### 后端技术栈
- Spring Boot：主要业务逻辑和API服务
- Python：机器学习模型训练和预测服务
- Neo4j：图数据库存储专利关系网络
- PyTorch：深度学习框架，用于RGCN模型
- scikit-learn：机器学习库，用于聚类算法

### 前端技术栈
- Vue.js：前端框架
- Element Plus：UI组件库
- ECharts：数据可视化
- Vuex：状态管理
- Vue Router：路由管理

## 系统要求
- JDK 11+
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
│   ├── api/              # Python API服务
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
└── README.md            # 项目说明
```

## 安装说明
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

# 安装前端依赖
cd ../frontend
npm install
```

## 启动服务
1. 启动后端服务
```bash
# 启动Spring Boot服务
cd backend
# Windows系统使用:
mvnw spring-boot:run
# Linux/Mac系统使用:
./mvnw spring-boot:run

# 启动Python服务
cd ../src/api
python app.py
```

2. 启动前端服务（开发模式）
```bash
cd frontend
npm run serve
```

## API文档
主要API接口：
- `GET /api/patents/random`: 获取随机专利列表
- `GET /api/patents/{patentId}/similar`: 获取相似专利
- `POST /api/patents/cluster`: 专利聚类分析

详细API文档请参考Swagger文档：`http://localhost:8080/swagger-ui.html`

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