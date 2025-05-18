import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Optional, Tuple
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter
import os

class RGCN(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, name2id: dict, no2label: dict, no_of_pubno: list, relation2id: dict, config_path: str = "config/config.yaml"):
        """初始化RGCN模型"""
        super().__init__()
        self.logger = logging.getLogger(__name__)#通信
        self.config = self._load_config(config_path)#配置
        # 数据层
        self.name2id         
        self.no2label
        self.no_of_pubno
        self.relation2id
        
        
        # 模型参数
        self.hidden_dim = self.config['hidden_dim']
        self.num_bases = self.config['num_bases']
        self.dropout = self.config['dropout']
        
        # 模型层
        self.embedding = nn.Embedding(num_nodes, self.hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.conv1 = RGCNConv(self.hidden_dim, self.hidden_dim,
                             num_relations, num_bases=self.num_bases, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        
        self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim,
                             num_relations, num_bases=self.num_bases, bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        
        # TensorBoard
        self.writer = SummaryWriter('runs/rgcn_training')

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['model']['rgcn']
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        if x is None:
            x = self.embedding.weight
            
        h = self.conv1(x, edge_index, edge_type)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h2 = self.conv2(h, edge_index, edge_type)
        h2 = self.norm2(h2)
        
        out = h2 + self.res_scale * h
        return out

    def train_model(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                   optimizer: torch.optim.Optimizer, num_epochs: int,
                   early_stopping_patience: int = 5) -> None:
        """训练模型"""
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            
            # 前向传播
            out = self(edge_index, edge_type)
            
            # 计算损失
            num_edges = edge_index.shape[1]
            src = edge_index[0]
            dst = edge_index[1]
            
            # 负采样
            perm = torch.randperm(num_edges)
            neg_dst = dst[perm]
            
            # 计算相似度
            h_src = out[src]
            h_pos = out[dst]
            h_neg = out[neg_dst]
            
            sim_pos = F.cosine_similarity(h_src, h_pos, dim=1)
            sim_neg = F.cosine_similarity(h_src, h_neg, dim=1)
            
            # InfoNCE损失
            temperature = 0.5
            logits = torch.exp(sim_pos / temperature) / (
                torch.exp(sim_pos / temperature) + torch.exp(sim_neg / temperature))
            loss = -torch.log(logits + 1e-8).mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录训练信息
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # 早停
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                self.save_model('models/best_rgcn.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

    def save_model(self, path: str) -> None:
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_weight': self.embedding.weight
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.embedding.weight = nn.Parameter(checkpoint['embedding_weight'])
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_embeddings(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """获取实体嵌入"""
        self.eval()
        with torch.no_grad():
            return self(edge_index, edge_type) 

    # 根据专利数据字典建立专利数据实体的字典
    def build_new_nodes_dict(self, new_patent):
        new_nodes_dict = {}
        for key, value in new_patent.items():
            if key == "PubNo-公开号":
                new_nodes_dict[value] = "PubNo"
            elif key == "Author-作者":
                authors = value.split(";")
                for author in authors:
                    new_nodes_dict[author.strip()] = "Author"
            elif key == "CLC-中国分类号":
                clcs = value.split(";")
                for clc in clcs:
                    new_nodes_dict[clc.strip()] = "CLC"
                stopwords = set(['本发明', '一种', '的', '用于','反应','应用','任意','所述','作为','在于','II','所示','特征','以下'])
        # 提取关键词
        keywords = []
        stopwords = set(['本发明', '一种', '的', '用于','反应','应用','任意','所述','作为','在于','II','所示','特征','以下'])
        # 提取标题的第1个关键词添加到列表中
        title_keyword = [word for word in jieba.analyse.extract_tags(new_patent['Title-题名'], topK=3) if word not in stopwords]
        keywords.append(title_keyword[0])  # 将标题的第一个关键词添加到列表中
        # 提取摘要的第1、2个关键词添加到列表中
        summary_keyword = [word for word in jieba.analyse.extract_tags(new_patent['Summary-摘要'], topK=10) if word not in stopwords and word not in title_keyword[0]]
        keywords.append(summary_keyword[0]) 
        keywords.append(summary_keyword[1]) 
        # 提取主权项的第1、2个关键词添加到列表中
        claims_keyword = [word for word in jieba.analyse.extract_tags(new_patent['Claims-主权项'], topK=20) if word not in stopwords and word not in title_keyword[0] and word not in summary_keyword] 
        keywords.append(claims_keyword[0])
        keywords.append(claims_keyword[1])
        #print(keywords)
        for keyword in keywords:
            new_nodes_dict[keyword] = "Keyword"
        return new_nodes_dict

    # 根据专利数据-实体-字典建立新节点的映射
    def add_new_node_index_map(self, new_nodes_dict):
        num_new_nodes = 0

        for node_name, node_label in new_nodes_dict.items():
            if node_name not in self.name2id:
                new_id = len(self.name2id)
                self.name2id[node_name] = new_id
                self.no2label[new_id] = node_label
                #print(f"添加新节点 {node_name}:{name2id[node_name]}，类型 {no2label[new_id]}，索引 {new_id}")
                num_new_nodes += 1
                if node_label == "PubNo":
                    self.no_of_pubno.append(new_id)
            else:
                #print(f"节点 {node_name}:{name2id[node_name]} 已存在，跳过添加。")
                continue

        return num_new_nodes

    #KeywordIs, CLCIs, AuthorIs
    # 根据专利数据-实体-字典建立边的映射, 返回新的边索引和边类型
    def BuildAdd_new_edges(self.new_nodes_dict):
        new_edges = []
        # 找出 pubno 节点（假设只有一个）
        pubno_nodes = [k for k, v in new_nodes_dict.items() if v == "PubNo"]
        if len(pubno_nodes) != 1:
            raise ValueError("应当有且仅有一个 pubno 节点")
        pubno = pubno_nodes[0]

        for entity, entity_type in new_nodes_dict.items():
            if entity == pubno:
                continue
            if entity_type == "Keyword":
                new_edges.append((pubno, "KeywordIs", entity))
            elif entity_type == "CLC":
                new_edges.append((pubno, "CLCIs", entity))
            elif entity_type == "Author":
                new_edges.append((pubno, "AuthorIs", entity))

        edge_index = []
        edge_type = []
        for edge in new_edges:# edge: (head, rel, tail)
            # 更新边索引和边类型
            edge_index.append([self.name2id[edge[0]], self.name2id[edge[2]]])
            edge_type.append(self.relation2id.setdefault(edge[1], len(self.relation2id)))
    
        # 所有关系的[[headset],[tailset]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()      # shape: [2, num_edges]
        # 所有关系的[rel_set]
        edge_type = torch.LongTensor(edge_type)        # shape: [num_edges]
    
        return edge_index, edge_type

    def new_patent_cluster(self, edge_index_new, edge_type_new, num_new_nodes):
        # 1.拼接新旧边
        self.edge_index = torch.cat([self.edge_index, edge_index_new], dim=1)
        self.edge_type = torch.cat([self.edge_type, edge_type_new], dim=0)

        # 原 embedding: shape [n, hidden_dim]
        old_embedding = self.embedding
        old_weight = old_embedding.weight.data  # 保存旧的参数
        n, hidden_dim = old_weight.shape

        # 创建新的 embedding
        new_embedding = nn.Embedding(n + num_new_nodes, hidden_dim)

        # 复制旧的
        with torch.no_grad():
            new_embedding.weight[:n] = old_weight  # 保留旧参数
            nn.init.xavier_uniform_(new_embedding.weight[n:])  # 初始化新节点

        # 替换掉原模型的 embedding
        self.embedding = new_embedding

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.train()

        for epoch in range(5):
            out = self(edge_index, edge_type)  # shape: [num_nodes, hidden_dim]

            num_edges = edge_index.shape[1]
            loss = 0

            # ---------------- 正边 (真实邻接边) ----------------
            src = edge_index[0]
            dst = edge_index[1]

            # ---------------- 负边 (随机负采样) ----------------
            # 随机打乱 dst 得到负例（也可以更复杂负采样）
            perm = torch.randperm(num_edges)
            neg_dst = dst[perm]

            # 取出嵌入向量
            h_src = out[src]
            h_pos = out[dst]
            h_neg = out[neg_dst]

            # 计算 cosine 相似度
            sim_pos = F.cosine_similarity(h_src, h_pos, dim=1)   # shape: [num_edges]
            sim_neg = F.cosine_similarity(h_src, h_neg, dim=1)   # shape: [num_edges]

            # InfoNCE 对比损失
            logits = torch.exp(sim_pos / temperature) / (torch.exp(sim_pos / temperature) + torch.exp(sim_neg / temperature))
            loss = -torch.log(logits + 1e-8).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        return

    # 直接调用的函数，接受新专利数据字典返回同类型的专利推荐
    def new_patent_samecluster_recommend(self, new_patent, num_clusters, y_true): 
        new_nodes_dict = self.build_new_nodes_dict(new_patent)
        num_new_nodes = self.add_new_node_index_map(new_nodes_dict)
        edge_index_new, edge_type_new = self.BuildAdd_new_edges(new_nodes_dict)
        self.new_patent_cluster(edge_index_new, edge_type_new, num_new_nodes)
        # 设定
        self.eval()
        with torch.no_grad():
            all_embeddings = self(edge_index, edge_type)  # 输出 shape: [num_entities, hidden_dim]

        # 假设你要对某一类实体进行聚类（已知其索引）
        selected_idx = torch.tensor(self.no_of_pubno)  # 指定要聚类的实体索引 - 专利类
        selected_embeddings = all_embeddings[selected_idx]

        # ---------------- KMeans 聚类 ----------------
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        y_pred = kmeans.fit_predict(selected_embeddings.cpu().numpy())
        y_pre_pred = y_pred[:len(y_true)]
        acc, row_ind, col_ind = clustering_accuracy(y_true, y_pre_pred)
        target_label = col_ind[y_pred[-1]]  # 获取目标标签
        filtered_df = df[df["Label-标签"] == target_label].sample(n=10, random_state=42) # 随机推荐10个专利
        return filtered_df, target_label
