import os
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from typing import List, Dict, Any

class PatentMLModel:
    """专利机器学习模型封装类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)
        
        # 加载训练好的聚类模型
        model_path = os.path.join(os.path.dirname(__file__), '../../models/trained_models/patent_cluster_model.pkl')
        if os.path.exists(model_path):
            self.cluster_model = joblib.load(model_path)
        else:
            self.cluster_model = KMeans(n_clusters=5, random_state=42)
    
    def extract_features(self, text: str) -> np.ndarray:
        """使用BERT提取文本特征"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的输出作为文本特征
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return features
    
    def cluster_patents(self, patents: List[Dict[str, Any]], n_clusters: int = 5) -> Dict[str, Any]:
        """对专利进行聚类分析"""
        # 提取特征
        features = []
        for patent in patents:
            text = f"{patent['title']} {patent['abstract']}"
            feature = self.extract_features(text)
            features.append(feature[0])  # 取第一个样本，因为batch_size=1
        
        features = np.array(features)
        
        # 如果需要重新训练聚类模型
        if len(patents) < n_clusters:
            n_clusters = len(patents)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_model.fit(features)
        
        # 获取聚类结果
        labels = self.cluster_model.labels_
        
        # 组织返回结果
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'patent_id': patents[i]['patent_id'],
                'title': patents[i]['title'],
                'distance': float(np.linalg.norm(features[i] - self.cluster_model.cluster_centers_[label]))
            })
        
        # 对每个簇内的专利按照到中心点的距离排序
        for label in clusters:
            clusters[label].sort(key=lambda x: x['distance'])
        
        return {
            'n_clusters': n_clusters,
            'clusters': clusters,
            'cluster_centers': self.cluster_model.cluster_centers_.tolist()
        }
    
    def find_similar_patents(self, target_patent: Dict[str, Any], patent_pool: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """查找相似专利"""
        # 提取目标专利特征
        target_text = f"{target_patent['title']} {target_patent['abstract']}"
        target_feature = self.extract_features(target_text)
        
        # 提取专利池中所有专利的特征
        pool_features = []
        for patent in patent_pool:
            text = f"{patent['title']} {patent['abstract']}"
            feature = self.extract_features(text)
            pool_features.append(feature[0])
        
        pool_features = np.array(pool_features)
        
        # 计算相似度（使用欧氏距离）
        distances = np.linalg.norm(pool_features - target_feature, axis=1)
        
        # 获取最相似的专利
        top_indices = np.argsort(distances)[:top_k]
        
        similar_patents = []
        for idx in top_indices:
            similar_patents.append({
                'patent_id': patent_pool[idx]['patent_id'],
                'title': patent_pool[idx]['title'],
                'similarity_score': float(1 / (1 + distances[idx]))  # 将距离转换为相似度分数
            })
        
        return similar_patents 