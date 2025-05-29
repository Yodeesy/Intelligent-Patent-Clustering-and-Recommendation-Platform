import pandas as pd
import torch
import sys
import os
import joblib

# === 设置路径 ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# === 加载缓存数据 ===
name2id = joblib.load('cache/name2id.pkl')               # 专利名称映射为节点 ID
no2label = joblib.load('cache/no2label.pkl')             # pub_no -> label
no_of_pubno = joblib.load('cache/no_of_pubno.pkl')       # pub_no -> 节点 ID
edge_index = torch.tensor(joblib.load('cache/edge_index.pkl'), dtype=torch.long)
edge_type = torch.tensor(joblib.load('cache/edge_type.pkl'), dtype=torch.long)
relation2id = joblib.load('cache/relation2.pkl')

# === 加载模型 ===
from models.rgcn import RGCN

model = RGCN(
    num_nodes=len(name2id),
    num_relations=len(relation2id),
    name2id=name2id,
    no2label=no2label,
    no_of_pubno=no_of_pubno,
    relation2id=relation2id,
    config_path='../src/configs/config_rgcn.yaml'
)

# 加载权重
state_dict = torch.load('../models/best_rgcn.pt')  # 这里是 OrderedDict
model.load_state_dict(state_dict['model_state_dict'])
model.eval()

# === 获取节点嵌入 ===
with torch.no_grad():
    embeddings = model.get_embeddings(edge_index, edge_type)  # shape: [num_nodes, emb_dim]

# === 读取专利数据 ===
data = pd.read_csv('processed/Patent.csv', encoding='gbk')

# === 获取每条专利的节点 ID（以 pub_no 为主）===
# 优先使用 no_of_pubno 来查嵌入，因为它直接以 pub_no 为 key
no_of_pubno = {pub_no: idx for idx, pub_no in enumerate(no_of_pubno)}
def get_node_id(row):
    pub_no = row['pub_no']
    return no_of_pubno.get(pub_no, None)

data['node_id'] = data.apply(get_node_id, axis=1)

# === 提取嵌入并存入新列 ===
def get_embedding(node_id):
    if node_id is None or pd.isna(node_id):
        return [0.0] * embeddings.shape[1]
    else:
        return embeddings[int(node_id)].tolist()

data['embedding'] = data['node_id'].apply(get_embedding)

# === 保留指定字段并导出 CSV ===
keep_columns = [
    'SrcDatabase', 'CLC', 'classification', 'pub_no', 'pub_date',  'title', 'author', 'applicant', 'province', 
    'abstract', 'claims', 'embedding'
]

data = data[keep_columns]
data.to_csv('processed/Patent_with_embedding.csv', index=False)
print("成功生成含嵌入向量的专利数据表：processed/Patent_with_embedding.csv")
