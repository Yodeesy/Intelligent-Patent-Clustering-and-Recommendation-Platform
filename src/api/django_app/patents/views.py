from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Patent
from .ml_models import PatentMLModel
from django.shortcuts import get_object_or_404

class PatentViewSet(viewsets.ModelViewSet):
    """专利API视图集"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ml_model = PatentMLModel()
    
    @action(detail=False, methods=['post'])
    def cluster(self, request):
        """专利聚类分析"""
        try:
            patents = request.data.get('patents', [])
            n_clusters = request.data.get('n_clusters', 5)
            
            if not patents:
                return Response(
                    {'error': '没有提供专利数据'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 执行聚类分析
            results = self.ml_model.cluster_patents(patents, n_clusters)
            
            return Response(results)
            
        except Exception as e:
            return Response(
                {'error': f'聚类分析失败: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def similar(self, request, pk=None):
        """查找相似专利"""
        try:
            # 获取目标专利
            target_patent = get_object_or_404(Patent, patent_id=pk)
            
            # 获取专利池（这里简单处理，实际应用中可能需要分页或其他筛选）
            patent_pool = Patent.objects.exclude(patent_id=pk)[:100]
            
            # 转换为字典格式
            target_dict = {
                'patent_id': target_patent.patent_id,
                'title': target_patent.title,
                'abstract': target_patent.abstract
            }
            
            pool_dicts = [
                {
                    'patent_id': p.patent_id,
                    'title': p.title,
                    'abstract': p.abstract
                }
                for p in patent_pool
            ]
            
            # 查找相似专利
            similar_patents = self.ml_model.find_similar_patents(
                target_dict,
                pool_dicts,
                top_k=request.query_params.get('top_k', 5)
            )
            
            return Response(similar_patents)
            
        except Patent.DoesNotExist:
            return Response(
                {'error': '专利不存在'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'查找相似专利失败: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 