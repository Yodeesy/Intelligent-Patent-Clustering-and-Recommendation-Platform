import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import logging
import yaml
import os
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

class PatentClustering:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化聚类模型"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.kmeans = None
        self.tsne = None

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['model']['clustering']
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise

    def fit(self, embeddings: torch.Tensor, n_clusters: int = None) -> None:
        """训练聚类模型"""
        if n_clusters is None:
            n_clusters = self.config['num_clusters']
            
        # 转换为numpy数组
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
            
        # 训练KMeans
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['random_state']
        )
        self.kmeans.fit(embeddings)
        
        # 训练t-SNE
        self.tsne = TSNE(
            n_components=2,
            random_state=self.config['random_state']
        )
        self.tsne_result = self.tsne.fit_transform(embeddings)

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """预测聚类标签"""
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
            
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
            
        return self.kmeans.predict(embeddings)

    def visualize(self, labels: np.ndarray = None, save_path: str = None) -> None:
        """可视化聚类结果"""
        if self.tsne_result is None:
            raise ValueError("No t-SNE results available!")
            
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            self.tsne_result[:, 0],
            self.tsne_result[:, 1],
            c=labels if labels is not None else self.kmeans.labels_,
            cmap='tab10'
        )
        plt.colorbar(scatter)
        plt.title("t-SNE Visualization of Patent Clusters")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def evaluate(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
        """评估聚类结果"""
        def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = y_true.astype(np.int64)
            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(len(y_pred)):
                w[y_pred[i], y_true[i]] += 1
            row_ind, col_ind = linear_sum_assignment(w.max() - w)
            return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(y_pred)

        acc = clustering_accuracy(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        metrics = {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari
        }
        
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        return metrics

    def get_cluster_centers(self) -> np.ndarray:
        """获取聚类中心"""
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
        return self.kmeans.cluster_centers_

    def get_similar_patents(self, query_embedding: torch.Tensor,
                          all_embeddings: torch.Tensor,
                          patent_ids: List[str],
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """获取与查询专利最相似的专利"""
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(all_embeddings, torch.Tensor):
            all_embeddings = all_embeddings.cpu().numpy()
            
        # 计算余弦相似度
        similarities = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(patent_ids[i], similarities[i]) for i in top_indices]

    def save_model(self, path: str) -> None:
        """保存模型"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'kmeans': self.kmeans,
            'tsne': self.tsne,
            'tsne_result': self.tsne_result
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        import joblib
        try:
            model_dict = joblib.load(path)
            self.kmeans = model_dict['kmeans']
            self.tsne = model_dict['tsne']
            self.tsne_result = model_dict['tsne_result']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise 