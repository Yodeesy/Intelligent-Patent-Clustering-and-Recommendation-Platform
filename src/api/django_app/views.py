from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import torch
import os
import logging
from src.database.neo4j_manager import Neo4jManager
from src.models.rgcn import RGCN
from src.models.clustering import PatentClustering

logger = logging.getLogger(__name__)

class PatentClusteringView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = Neo4jManager()
        self.rgcn = None
        self.clustering = None
        self._load_models()

    def _load_models(self):
        """加载预训练模型"""
        try:
            # 加载RGCN模型
            name_to_id, no_to_label = self.db.get_node_index_map()
            edge_index, edge_type, relation_to_id = self.db.get_edges(name_to_id)
            
            self.rgcn = RGCN(
                num_nodes=len(name_to_id),
                num_relations=len(relation_to_id)
            )
            self.rgcn.load_model('models/best_rgcn.pt')
            
            # 加载聚类模型
            self.clustering = PatentClustering()
            self.clustering.load_model('models/clustering.joblib')
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def get(self, request):
        """获取随机专利列表"""
        try:
            size = int(request.query_params.get('size', 100))
            patents = self.db.get_random_patents(size)
            return Response(patents)
        except Exception as e:
            logger.error(f"Error in get_random_patents: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request):
        """处理专利聚类请求"""
        try:
            # 获取选中的专利ID
            patent_ids = request.data.get('patent_ids', [])
            if not patent_ids:
                return Response(
                    {"error": "No patents selected"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 获取专利嵌入
            name_to_id, no_to_label = self.db.get_node_index_map()
            edge_index, edge_type, _ = self.db.get_edges(name_to_id)
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            
            # 获取所有实体的嵌入
            all_embeddings = self.rgcn.get_embeddings(edge_index, edge_type)
            
            # 获取选中专利的嵌入
            selected_indices = [name_to_id[pid] for pid in patent_ids]
            selected_embeddings = all_embeddings[selected_indices]
            
            # 预测聚类
            cluster_labels = self.clustering.predict(selected_embeddings)
            
            # 获取相似专利
            results = []
            for i, pid in enumerate(patent_ids):
                cluster_id = cluster_labels[i]
                similar_patents = self.clustering.get_similar_patents(
                    selected_embeddings[i],
                    all_embeddings,
                    list(name_to_id.keys()),
                    top_k=10
                )
                
                # 获取相似专利的详细信息
                similar_patents_info = []
                for similar_pid, similarity in similar_patents:
                    if similar_pid != pid:  # 排除自身
                        patent_info = self.db.get_patent_by_id(similar_pid)
                        patent_info['similarity'] = float(similarity)
                        similar_patents_info.append(patent_info)
                
                results.append({
                    'patent_id': pid,
                    'cluster_id': int(cluster_id),
                    'similar_patents': similar_patents_info
                })
            
            return Response({
                'results': results,
                'total_clusters': len(set(cluster_labels))
            })
            
        except Exception as e:
            logger.error(f"Error in process_clustering: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 