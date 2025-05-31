from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import joblib
import csv

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# ------------------ 数据与模型加载 ------------------


# 加载预处理后的专利数据（包括 embedding 字符串）
patents_data = pd.read_csv('../data/processed/Patent_with_embedding.csv')

# 将字符串形式的 embedding 转为 float32 的 numpy array
def str_to_array_float32(s):
    return np.array(eval(s), dtype=np.float32)

patents_data['embedding'] = patents_data['embedding'].apply(str_to_array_float32)

# 加载聚类模型
kmeans_dict = joblib.load('../models/clustering.joblib')
kmeans: KMeans = kmeans_dict['kmeans']

# 构造 embedding 矩阵用于聚类预测
embeddings = np.stack(patents_data['embedding'].values)

# 为每个专利预测其所属聚类
patents_data['cluster'] = kmeans.predict(embeddings)

# ------------------ 工具函数 ------------------

def compute_cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ------------------ 接口定义 ------------------

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """读取 Patent_with_keys.csv 文件的前 100 行并返回"""
    examples = []
    try:
        with open('../data/processed/Patent_with_keys.csv', 'r', encoding='gbk') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i < 100:
                    examples.append(row)
                else:
                    break
        return jsonify(examples)
    except FileNotFoundError:
        return jsonify({"error": "Patent_with_keys.csv 文件未找到"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    """根据多个字段推荐相似专利"""
    params = {
        'author': request.args.get('author', '').strip(),
        'applicant': request.args.get('applicant', '').strip(),
        'title': request.args.get('title', '').strip(),
        'province': request.args.get('province', '').strip(),
        'pub_no': request.args.get('pub_no', '').strip(),
        'pub_date': request.args.get('pub_date', '').strip(),
        'CLC': request.args.get('CLC', '').strip(),
    }

    filtered = patents_data.copy()
    for key, value in params.items():
        if value:
            filtered = filtered[filtered[key].astype(str).str.contains(value, case=False, na=False)]

    if filtered.empty:
        return jsonify({"code": 404, "message": "未找到符合条件的专利"})

    query_vec = filtered.iloc[0]['embedding']
    results = patents_data.copy()
    results['similarity'] = results['embedding'].apply(lambda x: compute_cosine_similarity(x, query_vec))

    top_results = results.sort_values('similarity', ascending=False).head(10)

    return jsonify({
        "code": 200,
        "message": "推荐成功",
        "data": top_results[[
            'SrcDatabase', 'CLC', 'classification', 'pub_no', 'pub_date',  'title', 'author', 'applicant', 'province', 
         'abstract', 'claims', 'similarity'
        ]].to_dict('records')
    })


@app.route('/api/cluster', methods=['GET'])
def cluster_view():
    """聚类查看接口：返回某个聚类下的所有专利"""
    try:
        cid = int(request.args.get('cluster_id'))
    except (TypeError, ValueError):
        return jsonify({"code": 400, "message": "请提供有效的 cluster_id"})

    cluster_data = patents_data[patents_data['cluster'] == cid]
    if cluster_data.empty:
        return jsonify({"code": 404, "message": f"聚类 {cid} 中无专利或不存在"})

    return jsonify({
        "code": 200,
        "message": f"聚类 {cid} 的专利列表",
        "data": cluster_data[['pub_no', 'title', 'abstract']].to_dict('records')
    })


@app.route('/api/clusters', methods=['GET'])
def cluster_summary():
    """聚类总览接口：返回每个聚类中包含的专利数"""
    summary = patents_data['cluster'].value_counts().sort_index().to_dict()
    return jsonify({
        "code": 200,
        "message": "聚类分布统计",
        "data": summary
    })


# ------------------ 主程序入口 ------------------

if __name__ == '__main__':
    app.run(debug=True)
