# -*- coding: utf-8 -*-
from neo4j import GraphDatabase
import pandas as pd
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from models.rgcn import RGCN
import logging
from torch.optim import Adam
import yaml

attributes = ['Title', 'SrcDatabase','CountryName',  'PubTime', 'Summary', 'Claims']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ======== 捕获实体及其属性特征向量函数 =========
def get_node_index_map(tx):
    #attr_str = ", ".join([f"n.{attr} AS {attr}" for attr in attributes])
    query = f"MATCH (n) RETURN n.name AS name, labels(n) AS label"#, {attr_str}"
    result = tx.run(query)
    name_to_id = {}
    no_to_label = {}
    for i, record in enumerate(result):
        name_to_id[record["name"]] = i
        no_to_label[i] = record["label"]
    return name_to_id, no_to_label

# ======== 捕获边函数 =========
def get_edges(tx, name_to_id, relation_to_id):
    result = tx.run("MATCH (a)-[r]->(b) RETURN a.name AS head, type(r) AS rel, b.name AS tail")
    edge_index = []
    edge_type = []
    for record in result:
        head = name_to_id[record["head"]]
        tail = name_to_id[record["tail"]]
        rel = relation_to_id.setdefault(record["rel"], len(relation_to_id))  # 编号从0开始
        edge_index.append([head, tail])
        edge_type.append(rel)
    return edge_index, edge_type, relation_to_id

import os
import pickle
import pandas as pd

def load_data():
    """从缓存文件中加载专利图数据"""
    
    # 1. 注释掉原始的 Neo4j 连接逻辑 =========
    """
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Aa123456"

    driver = GraphDatabase.driver(uri, auth=(username, password))

    edge_index = []
    edge_type = []
    name2id = {}
    no2label = {}
    relation2id = {}
    with driver.session(database="final") as session:
        name2id, no2label = session.execute_read(get_node_index_map)
        edge_index, edge_type, relation2id = session.execute_read(get_edges, name2id, relation2id)

    driver.close()
    """

    # 2. 使用缓存加载数据 =========
    cache_path = os.path.join("data", "cache")

    with open(os.path.join(cache_path, 'name2id.pkl'), 'rb') as f:
        name2id = pickle.load(f)
    with open(os.path.join(cache_path, 'no2label.pkl'), 'rb') as f:
        no2label = pickle.load(f)
    with open(os.path.join(cache_path, 'relation2.pkl'), 'rb') as f:
        relation2id = pickle.load(f)
    with open(os.path.join(cache_path, 'no_of_pubno.pkl'), 'rb') as f:
        no_of_pubno = pickle.load(f)
    with open(os.path.join(cache_path, 'edge_index.pkl'), 'rb') as f:
        edge_index = pickle.load(f)
    with open(os.path.join(cache_path, 'edge_type.pkl'), 'rb') as f:
        edge_type = pickle.load(f)

    # 3. 加载标签信息 =========
    pubno_vals = [k for k, v in name2id.items() if v in no_of_pubno]
    df = pd.read_csv("data/processed/Train_Patent.csv")

    y_true = []

    for i in pubno_vals:
        label_value = df.loc[df['PubNo-公开号'] == i, 'Label-标签'].values
        if label_value.size > 0:
            y_true.append(label_value[0])
        else:
            y_true.append(None)  # 若没有对应标签，可选择填 None 或跳过

    return name2id, no2label, no_of_pubno, relation2id, edge_index, edge_type, y_true


def train():
    """训练模型"""
    # 加载配置
    config = load_config()
    model_config = config['model']['rgcn']
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载数据
    data, num_nodes = load_data()
    data = data.to(device)
    
    # 初始化模型
    model = RGCN(
        num_nodes=num_nodes,
        num_relations=1,  # 目前只有引用关系一种类型
        config_path="config/config.yaml"
    ).to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=model_config['learning_rate'])
    
    # 训练模型
    logger.info("Starting training...")
    model.train_model(
        edge_index=data.edge_index,
        edge_type=data.edge_type,
        optimizer=optimizer,
        num_epochs=model_config['num_epochs'],
        early_stopping_patience=model_config['early_stopping_patience']
    )
    
    logger.info("Training completed!")
    
    # 保存最终模型
    model.save_model('models/final_rgcn.pt')
    
    return model

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    try:
        model = train()
        logger.info("Model training successful!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 
