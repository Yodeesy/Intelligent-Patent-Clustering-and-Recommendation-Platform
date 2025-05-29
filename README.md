# Intelligent Patent Clustering and Recommendation Platform

## 一、项目简介

这是一个基于图神经网络和机器学习的智能专利聚类与推荐平台。该平台采用微服务架构，将业务逻辑和机器学习服务分离，提供专利文献的智能分析、聚类和推荐功能。

## 二、项目结构

```txt
patent/
├── data/
│   ├── processed/         # 处理后的数据
│   ├── cache/             # 缓存数据
│   └── build.py           # 构建图数据
│   └── make_embedding.py  # 生成专利嵌入向量
├── src/
│   ├── configs/           # 配置文件
│   │   ├── config_clustering.yaml  # 聚类模型配置
│   │   └── config_rgcn.yaml        # RGCN模型配置
│   └── models/            # 模型代码
│       ├── clustering.py  # 聚类模型
│       └── rgcn.py        # RGCN模型
├── frontend/
│   └── index.html         # 前端页面
├── backend/
│   └── app.py             # 后端接口
├── scripts/
│   └── pretrain.py        # 模型预训练脚本
└── models/                # 训练好的模型
```

## 三、环境配置

### 1. 安装依赖库

在项目根目录下，使用以下命令安装所需的 Python 库：

```bash
pip install -r requirements.txt
```

### 2. 配置文件说明

- **src/configs/config_clustering.yaml**：聚类模型的配置文件，包括聚类数量、KMeans 和 t-SNE 的参数等。
- **src/configs/config_rgcn.yaml：**RGCN 模型的配置文件，包括隐藏层维度、基矩量数量、训练轮数等参数。

## 四、数据准备

### 1. 数据处理

运行data/build.py脚本，将原始的专利数据处理为图数据，并保存到`data/cache`目录下：

```bash
python data/build.py
```

### 2. 生成嵌入向量

运行data/make_embedding.py脚本，使用训练好的 RGCN 模型为每条专利生成嵌入向量，并保存为 CSV 文件：

```bash
python data/make_embedding.py
```

## 五、模型训练

运行scripts/pretrain.py脚本，进行 RGCN 模型和聚类模型的预训练：

```bash
python scripts/pretrain.py
```

训练完成后，模型将保存到`models`目录下。

## 六、启动服务

### 1. 启动后端服务

在`backend`目录下，运行app.py脚本启动 Flask 后端服务：

```bash
python backend/app.py
```

### 2. 打开前端页面

在浏览器中打开frontend/index.html文件，即可访问专利推荐系统的前端页面。

## 七、使用说明

### 1. 搜索相似专利

在前端页面的搜索框中输入作者、申请人、标题关键词等信息，点击 “搜索相似专利” 按钮，系统将返回与输入条件相似的专利列表。

### 2. 查看聚类总览

点击 “查看聚类总览” 按钮，系统将返回每个聚类中包含的专利数量。

### 3. 查看特定聚类下的专利列表

在 “输入聚类编号” 输入框中输入聚类编号，点击 “查看该聚类专利” 按钮，系统将返回该聚类下的所有专利列表。

## 八、注意事项

- 请确保输入的专利数据文件路径和格式正确，并且文件编码为`gbk`。
- 在训练模型时，可能需要根据实际情况调整配置文件中的参数，以获得更好的性能。
- 如果在运行过程中出现错误，请检查日志文件中的错误信息，并根据提示进行相应的处理。

--------

## 贡献指南

欢迎提交Issue和Pull Request。在提交PR之前，请确保：

1. 代码符合项目的编码规范
2. 添加必要的测试用例
3. 更新相关文档

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。