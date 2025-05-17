import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_manager import Neo4jManager
from src.models.rgcn import RGCN
from src.models.clustering import PatentClustering

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pretrain():
    """预训练模型"""
    try:
        # 1. 连接数据库并获取数据
        logger.info("Connecting to database...")
        db = Neo4jManager()
        
        # 2. 获取图数据
        logger.info("Getting graph data...")
        name_to_id, no_to_label = db.get_node_index_map()
        edge_index, edge_type, relation_to_id = db.get_edges(name_to_id)
        
        # 3. 准备训练数据
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        # 4. 初始化RGCN模型
        logger.info("Initializing RGCN model...")
        model = RGCN(
            num_nodes=len(name_to_id),
            num_relations=len(relation_to_id)
        )
        
        # 5. 训练RGCN模型
        logger.info("Training RGCN model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train_model(
            edge_index=edge_index,
            edge_type=edge_type,
            optimizer=optimizer,
            num_epochs=100,
            early_stopping_patience=5
        )
        
        # 6. 获取实体嵌入
        logger.info("Getting entity embeddings...")
        model.eval()
        with torch.no_grad():
            all_embeddings = model(edge_index, edge_type)
        
        # 7. 获取专利实体的索引
        patent_indices = [i for i, labels in no_to_label.items() if "PubNo" in labels]
        patent_embeddings = all_embeddings[patent_indices]
        
        # 8. 训练聚类模型
        logger.info("Training clustering model...")
        clustering = PatentClustering()
        clustering.fit(patent_embeddings)
        
        # 9. 保存模型
        logger.info("Saving models...")
        os.makedirs('models', exist_ok=True)
        model.save_model('models/best_rgcn.pt')
        clustering.save_model('models/clustering.joblib')
        
        # 10. 可视化结果
        logger.info("Visualizing results...")
        clustering.visualize(save_path='models/clustering_visualization.png')
        
        logger.info("Pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during pretraining: {str(e)}")
        raise

if __name__ == "__main__":
    pretrain() 