import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('Patent_with_keys.csv', encoding='utf-8')


entities = set()  # 使用集合来存储唯一的实体
triplets = [] # 提取的三元组将存储在这个列表中
attributes = [] # 用于存储属性

COLUMNS = df.columns # 获取所有列名

# 5种用于分类的关系
classified_columns = ['Author-作者','CLC-中图分类号',
           'Title_Keywords-名称关键字','Summary_Keywords-摘要关键字','Claims_Keywords-主权项关键字']  
#attribute_columns = ['Title-题名','SrcDatabase-来源库','CountryName-国省名称','PubTime-公开日期','Summary-摘要','Claims-主权项']

labelmapping = {
    'SrcDatabase-来源库':'SrcDatabase',
    'Author-作者':'Author',
    'Applicant-申请人':'Applicant',
    'Title-题名':'Title',
    'CountryName-国省名称':'CountryName',
    'PubNo-公开号':'PubNo',# uid
    'PubTime-公开日期':'PubTime',
    'Summary-摘要':'Summary',
    'Claims-主权项':'Claims',
    'CLC-中图分类号':'CLC',
    'Label-标签':'Label',
    'Title_Keywords-名称关键字':'Title_Keywords',
    'Summary_Keywords-摘要关键字':'Summary_Keywords',
    'Claims_Keywords-主权项关键字':'Claims_Keywords'
}

# 遍历每一行，合并 Title、Summary 和 Claims
for index,row in df.iterrows():
    for column in COLUMNS:
        if(column == 'PubNo-公开号'): # 公开号做唯一索引
            entities.add((row[column], labelmapping[column]))
        if(column in classified_columns):  # 若为关系点
            if(pd.isna(row[column])):  # 检查是否为空
                continue
            else:
                split_row = row[column].split(';')  # 分割字符串
                for i in split_row:
                    triplet = (row['PubNo-公开号'], labelmapping[column]+'Is', i.strip())  # 使用元组存储
                    
                    if(column=='Title_Keywords-名称关键字' or column=='Summary_Keywords-摘要关键字' or column=='Claims_Keywords-主权项关键字'):
                        entities.add((i.strip(), 'Keyword'))  # 添加关键词实体
                        triplet = (row['PubNo-公开号'], 'KeywordIs', i.strip())  # 使用元组存储
                    else:
                        entities.add((i.strip(), labelmapping[column]))  # 添加关键词实体
                    triplets.append(triplet)
                    #print(triplet)
                    #input()
        else:  # 若为属性点
            if(pd.isna(row[column])):  # 检查是否为空
                continue
            else:
                if(column != 'Label-标签'):
                    attributes.append([row['PubNo-公开号'], labelmapping['PubNo-公开号'], labelmapping[column], row[column].strip()])
                else:
                    attributes.append([row['PubNo-公开号'], labelmapping['PubNo-公开号'], labelmapping[column], row[column]])
                #print(attributes[-1])
                #input()

print(len(entities))
print(len(list(set(triplets))))
print(len(attributes))

from neo4j import GraphDatabase

# 1. 连接 Neo4j =========
uri = "bolt://localhost:7687"  # 修改为你的地址
username = "neo4j"
password = "Aa123456"     # 修改为你的密码

driver = GraphDatabase.driver(uri, auth=(username, password))

# ======== 2. 写入函数 =========

def create_entity(tx, name, label):
    query = f"MERGE (e:{label} {{name: $name}})"
    tx.run(query, name=name)

def create_relationship(tx, head, tail, relation):
    query = f"""
    MATCH (a {{name: $head}})
    MATCH (b {{name: $tail}})
    MERGE (a)-[r:{relation}]->(b)
    RETURN type(r)
    """
    tx.run(query, head=head, tail=tail)

def create_attribute(tx, name, label, attr_name, attr_value):
    query = f"""
    MATCH (e:{label} {{name: $name}})
    SET e.{attr_name} = $attr_value
    """
    tx.run(query, name=name, attr_value=attr_value)

with driver.session(database="final") as session:
    for name, label in entities:
        session.execute_write(create_entity, name, label)

    for head, relation, tail in triplets:
        session.execute_write(create_relationship, head, tail, relation)
    
    #for name, label, attr_name, attr_value in attributes:
        #session.execute_write(create_attribute, name, label, attr_name, attr_value)

driver.close()
