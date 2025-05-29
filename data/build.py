import pandas as pd
import pickle
from collections import defaultdict
import torch
import os

CSV_PATH = 'processed/Patent_with_keys.csv'
SAVE_DIR = 'cache/'
DELIMITER = ';'

# 字段配置
FIELDS = {
    'author': 'author',
    'applicant': 'applicant',
    'title_keys': 'keywords',
    'title': 'patent',   # 中心节点
}

# 实体和映射字典
name2id = {}
relation2id = {}
no2label = {}
no_of_pubno = {}
current_id = 0
edges = []

def get_or_add_id(name):
    global current_id
    if name not in name2id:
        name2id[name] = current_id
        current_id += 1
    return name2id[name]

df = pd.read_csv(CSV_PATH, encoding='gbk')

for _, row in df.iterrows():
    pubno = str(row['pub_no']).strip()
    title = str(row['title']).strip()
    label = int(row['label']) if not pd.isna(row['label']) else -1

    # 创建中心节点（专利）
    patent_id = get_or_add_id(title)
    no2label[pubno] = label
    no_of_pubno[pubno] = patent_id

    for field, entity_type in FIELDS.items():
        if entity_type == 'patent':
            continue
        if pd.isna(row[field]):
            continue

        items = str(row[field]).split(DELIMITER)
        for item in items:
            item = item.strip()
            if not item:
                continue

            entity_id = get_or_add_id(item)
            relation = f'{entity_type}->patent'
            if relation not in relation2id:
                relation2id[relation] = len(relation2id)
            rel_id = relation2id[relation]

            edges.append((entity_id, patent_id, rel_id))

# 创建 edge_index 和 edge_type
edge_index = [[], []]
edge_type = []
for src, tgt, rel in edges:
    edge_index[0].append(src)
    edge_index[1].append(tgt)
    edge_type.append(rel)

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_type = torch.tensor(edge_type, dtype=torch.long)

# 保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

with open(SAVE_DIR + 'edge_index.pkl', 'wb') as f:
    pickle.dump(edge_index, f)
with open(SAVE_DIR + 'edge_type.pkl', 'wb') as f:
    pickle.dump(edge_type, f)
with open(SAVE_DIR + 'name2id.pkl', 'wb') as f:
    pickle.dump(name2id, f)
with open(SAVE_DIR + 'relation2.pkl', 'wb') as f:
    pickle.dump(relation2id, f)
with open(SAVE_DIR + 'no2label.pkl', 'wb') as f:
    pickle.dump(no2label, f)
with open(SAVE_DIR + 'no_of_pubno.pkl', 'wb') as f:
    pickle.dump(no_of_pubno, f)

print("图构建完成！")
print("边数:", len(edge_type))
print("关系类型:", relation2id)
print("公开号数量:", len(no_of_pubno))
