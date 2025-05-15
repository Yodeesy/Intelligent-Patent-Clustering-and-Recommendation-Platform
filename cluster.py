from neo4j import GraphDatabase
import pandas as pd

# 1. 连接 Neo4j =========
uri = "bolt://localhost:7687"  # 修改为你的地址
username = "neo4j"
password = "Aa123456"     # 修改为你的密码

driver = GraphDatabase.driver(uri, auth=(username, password))
attributes = ['Title', 'SrcDatabase','CountryName',  'PubTime', 'Summary', 'Claims']

name2id = {}    # 所有实体的编号字典
no2label = {}   # 所有实体的类型字典
texts_features = []
# ======== 捕获实体及其属性特征向量函数 =========
def get_node_index_map(tx):
    attr_str = ", ".join([f"n.{attr} AS {attr}" for attr in attributes])
    query = f"MATCH (n) RETURN n.name AS name, labels(n) AS label, {attr_str}"
    result = tx.run(query)
    name_to_id = {}
    no_to_label = {}
    for i, record in enumerate(result):
        name_to_id[record["name"]] = i
        no_to_label[i] = record["label"]
    return name_to_id, no_to_label

# ======== 捕获边函数 =========
def get_edges(tx, name_to_id, relation_to_id):
    result = tx.run("MATCH (a)-[r]->(b) RETURN a.name AS head, type(r) AS rel, b.name AS tail")
    edge_index = []
    edge_type = []
    for record in result:
        head = name_to_id[record["head"]]
        tail = name_to_id[record["tail"]]
        rel = relation_to_id.setdefault(record["rel"], len(relation_to_id))  # 编号从0开始
        edge_index.append([head, tail])
        edge_type.append(rel)
    return edge_index, edge_type, relation_to_id

edge_index = []
edge_type = []
name2id = {}
relation2id = {}
with driver.session(database="final") as session:
    name2id, no2label = session.execute_read(get_node_index_map)
    edge_index, edge_type, relation2id = session.execute_read(get_edges, name2id, relation2id)

driver.close()

keys_with_pubno = [k for k, v in no2label.items() if "PubNo" in v]# 获取所有 PubNo实体点 的索引
pubno_labels = [k for k, v in name2id.items() if v in keys_with_pubno]# 获取所有 PubNo实体点 的key

y_true = []
# 读取 CSV 文件
df = pd.read_csv("Patent.csv")

for i in pubno_labels:
    # 查找对应 'PubNo' 的 'label' 值
    label_value = df.loc[df['PubNo-公开号'] == i, 'Label-标签'].values
    y_true.append(label_value[0])

for i in range(len(y_true)):
    value_to_find = keys_with_pubno[i]
    # 使用字典遍历来找到对应的 key
    keys = [k for k, v in name2id.items() if v == value_to_find]
    #print(keys,y_true[i])
#input()
# 实体点数量
node_num = len(name2id)
rel_num = len(relation2id)

import torch
import numpy as np
from torch import nn
from torch_geometric.nn import RGCNConv
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from torch.nn.init import xavier_normal_, xavier_uniform_
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

# ========== 训练 RGCN ==========
'''
class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=128, dropout=0.3):
        super(RGCN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=6)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=6)
        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = self.embedding.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # 🔥 加在第一层激活后
        x = self.conv2(x, edge_index, edge_type)
        return x
'''

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations,
                 hidden_dim=128, num_bases=6, dropout=0.3):
        super().__init__()

        # ⚡ 2.1  特征可选：预加载属性向量 / Glorot init 更稳定
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # ⚡ 2.2  RGCNConv 层加 bias=False，后面接 LayerNorm
        self.conv1 = RGCNConv(hidden_dim, hidden_dim,
                              num_relations, num_bases=num_bases, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # ⚡ 2.3  第二层 residual，避免过平滑
        self.conv2 = RGCNConv(hidden_dim, hidden_dim,
                              num_relations, num_bases=num_bases, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5))  # 可学习残差比例

        self.dropout = dropout

    def forward(self, edge_index, edge_type, x=None):
        if x is None:
            x = self.embedding.weight  # 默认用模型内部embedding
        h = self.conv1(x, edge_index, edge_type)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = self.conv2(h, edge_index, edge_type)
        h2 = self.norm2(h2)

        out = h2 + self.res_scale * h
        return out

# 所有关系的[[headset],[tailset]]
edge_index = torch.tensor(edge_index, dtype=torch.long).t()      # shape: [2, num_edges]
# 所有关系的[rel_set]
edge_type = torch.LongTensor(edge_type)        # shape: [num_edges]

print(node_num)
print(edge_index.shape)
print(edge_type.shape)
print(len(relation2id))

model = RGCN(num_nodes=node_num, num_relations=len(relation2id), hidden_dim=128, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
temperature = 0.5  # τ 超参数

for epoch in range(100):
    model.train()
    out = model(edge_index, edge_type)  # shape: [num_nodes, hidden_dim]

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

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 设定
model.eval()
with torch.no_grad():
    all_embeddings = model(edge_index, edge_type)  # 输出 shape: [num_entities, hidden_dim]

# 假设你要对某一类实体进行聚类（已知其索引）
selected_idx = torch.tensor(keys_with_pubno)  # 指定要聚类的实体索引，比如作者类、专利类
selected_embeddings = all_embeddings[selected_idx]

# 假设你有真实标签
y_true = np.array(y_true)  # shape: [num_selected_entities]

# ---------------- KMeans 聚类 ----------------
num_clusters = len(np.unique(y_true))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
y_pred = kmeans.fit_predict(selected_embeddings.cpu().numpy())

# ---------------- t-SNE 可视化 ----------------
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(selected_embeddings.cpu().numpy())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_pred, cmap='tab10', s=10)
plt.title("t-SNE Visualization of R-GCN Entity Embeddings")
plt.colorbar(scatter)
plt.show()

# ---------------- 聚类评估 ----------------
def clustering_accuracy(y_true, y_pred):
    # 用匈牙利算法对聚类标签重新编号
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(y_pred)

acc = clustering_accuracy(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)

print(f"ACC: {acc:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")


def add_new_node_index_map(new_nodes_dict):
    global name2id
    global no2label

    for node_name, node_label in new_nodes_dict.items():
        if node_name not in name2id:
            new_id = len(name2id)
            name2id[node_name] = new_id
            no2label[new_id] = node_label

#KeywordIs, CLCIs, AuthorIs
def add_new_edges(new_edges):
    edge_index_new = []
    pass

def get_patent(edge_index_old, edge_index_new, edge_type_old, edge_type_new, num_new_nodes):
    global model, name2id, relation2id
    #name2id.setdefault(record["rel"], len(relation_to_id))  # 编号从0开始
    # 1.拼接新旧边
    # edge_index_old: 旧边索引
    # edge_index_new: 新边索引
    edge_index = torch.cat([edge_index_old, edge_index_new], dim=1)
    edge_type = torch.cat([edge_type_old, edge_type_new], dim=0)

    # 2.更新网络
    # 2.1 更新 embedding 层
    # 替换 embedding 层（保持旧参数不变，新增的随机初始化）
    old_weight = model.embedding.weight.data
    num_old_nodes, hidden_dim = old_weight.size()

    new_embedding = nn.Embedding(num_new_nodes, hidden_dim)
    nn.init.xavier_uniform_(new_embedding.weight)
    new_embedding.weight.data[:num_old_nodes] = old_weight  # 保留旧节点的参数
    model.embedding = new_embedding.to(model.embedding.weight.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(10):
        out = model(edge_index, edge_type)  # shape: [num_nodes, hidden_dim]

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

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return 

