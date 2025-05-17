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
    def __init__(self, num_nodes: int, num_relations: int, config_path: str = "config/config.yaml"):
        """初始化RGCN模型"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
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