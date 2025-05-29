import os
import sys
import logging
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        
         # 获取当前脚本所在目录的上一级，即项目根目录
        ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        #临时读取cache
        cache_path = os.path.join(ROOT_DIR,"data", "cache")
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

        
        # 3. 准备训练数据
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        

        # 4. 初始化RGCN模型
        logger.info("Initializing RGCN model...")
        model = RGCN(
            num_nodes=len(name2id),
            num_relations=len(relation2id),  
            name2id=name2id,
            no2label=no2label,
            no_of_pubno=no_of_pubno,
            relation2id=relation2id,
            config_path=os.path.join(ROOT_DIR, "src", "configs", "config_rgcn.yaml")
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
        patent_indices = []
        for patent_no in no2label.keys():
            if patent_no in no_of_pubno:
                patent_indices.append(no_of_pubno[patent_no])
            else:
                print(f"[警告] 无法在 name2id 中找到专利号：{patent_no}")

        if len(patent_indices) == 0:
            raise ValueError("未能构造出专利实体索引数组，请确认 name2id 与 no2label 对应正确。")

        patent_embeddings = all_embeddings[torch.LongTensor(patent_indices)]
        
        # 8. 训练聚类模型
        logger.info("Training clustering model...")
        clustering = PatentClustering(
            config_path=os.path.join(ROOT_DIR, "src", "configs", "config_clustering.yaml"))
        clustering.fit(patent_embeddings)
        
        # 9. 保存模型
        logger.info("Saving models...")
        model.save_model(os.path.join(ROOT_DIR, "models", "best_rgcn.pt"))
        clustering.save_model(os.path.join(ROOT_DIR, "models", "clustering.joblib"))
        
        # 10. 可视化结果
        logger.info("Visualizing results...")
        clustering.visualize(save_path=os.path.join(ROOT_DIR, "models", "clustering_visualization.png"))
        
        logger.info("Pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during pretraining: {str(e)}")
        raise

if __name__ == "__main__":
    pretrain() 