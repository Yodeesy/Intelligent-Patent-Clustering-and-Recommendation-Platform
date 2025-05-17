import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from models.rgcn import RGCN
import logging
from torch.optim import Adam
import yaml

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

def load_data():
    """加载专利数据并构建图"""
    logger.info("Loading patent data...")
    
    # 加载处理后的数据
    patents_df = pd.read_csv('data/processed/Patent.csv')
    relations_df = pd.read_csv('data/processed/Patent_with_keys.csv')
    
    # 构建节点映射
    unique_patents = pd.concat([
        patents_df['patent_id'],
        relations_df['source_id'],
        relations_df['target_id']
    ]).unique()
    
    node_mapping = {node: idx for idx, node in enumerate(unique_patents)}
    num_nodes = len(node_mapping)
    
    # 构建边和边类型
    edge_index = []
    edge_type = []
    
    # 添加引用关系
    source_nodes = [node_mapping[pid] for pid in relations_df['source_id']]
    target_nodes = [node_mapping[pid] for pid in relations_df['target_id']]
    edge_index.extend(list(zip(source_nodes, target_nodes)))
    edge_type.extend([0] * len(source_nodes))  # 0表示引用关系
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    logger.info(f"Graph built with {num_nodes} nodes and {len(edge_type)} edges")
    
    return Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes
    ), num_nodes

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