import logging
from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Any
import yaml
import os

class Neo4jManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化Neo4j连接管理器"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.driver = self._create_driver()

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['database']['neo4j']
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise

    def _create_driver(self):
        """创建Neo4j驱动"""
        try:
            return GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['username'], self.config['password'])
            )
        except Exception as e:
            self.logger.error(f"Failed to create Neo4j driver: {str(e)}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def get_node_index_map(self) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        """获取节点索引映射"""
        with self.driver.session(database=self.config['database']) as session:
            result = session.run("""
                MATCH (n)
                RETURN n.name AS name, labels(n) AS label
            """)
            name_to_id = {}
            no_to_label = {}
            for i, record in enumerate(result):
                name_to_id[record["name"]] = i
                no_to_label[i] = record["label"]
            return name_to_id, no_to_label

    def get_edges(self, name_to_id: Dict[str, int]) -> Tuple[List[List[int]], List[int], Dict[str, int]]:
        """获取边信息"""
        with self.driver.session(database=self.config['database']) as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name AS head, type(r) AS rel, b.name AS tail
            """)
            edge_index = []
            edge_type = []
            relation_to_id = {}
            for record in result:
                head = name_to_id[record["head"]]
                tail = name_to_id[record["tail"]]
                rel = relation_to_id.setdefault(record["rel"], len(relation_to_id))
                edge_index.append([head, tail])
                edge_type.append(rel)
            return edge_index, edge_type, relation_to_id

    def get_random_patents(self, size: int) -> List[Dict[str, Any]]:
        """随机获取指定数量的专利"""
        with self.driver.session(database=self.config['database']) as session:
            result = session.run(f"""
                MATCH (p:PubNo)
                WITH p, rand() as r
                ORDER BY r
                LIMIT {size}
                RETURN p.name as pubno, p.Title as title, p.Summary as summary
            """)
            return [dict(record) for record in result]

    def get_patent_by_id(self, patent_id: str) -> Dict[str, Any]:
        """根据专利ID获取专利信息"""
        with self.driver.session(database=self.config['database']) as session:
            result = session.run("""
                MATCH (p:PubNo {name: $patent_id})
                RETURN p.name as pubno, p.Title as title, p.Summary as summary
            """, patent_id=patent_id)
            return dict(result.single())

    def get_similar_patents(self, patent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取与指定专利相似的专利"""
        with self.driver.session(database=self.config['database']) as session:
            result = session.run("""
                MATCH (p1:PubNo {name: $patent_id})-[:KeywordIs]->(k:Keyword)
                MATCH (p2:PubNo)-[:KeywordIs]->(k)
                WHERE p1 <> p2
                WITH p2, count(*) as common
                ORDER BY common DESC
                LIMIT $limit
                RETURN p2.name as pubno, p2.Title as title, p2.Summary as summary, common
            """, patent_id=patent_id, limit=limit)
            return [dict(record) for record in result]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 