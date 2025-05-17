import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List
import re
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """清理文本数据"""
    if pd.isna(text):
        return ""
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', str(text))
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_patent_data(input_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """处理专利数据"""
    logger.info(f"Reading data from {input_file}")
    
    # 读取Excel文件
    df = pd.read_excel(input_file)
    
    # 打印列名
    logger.info(f"Columns in Excel file: {df.columns.tolist()}")
    
    # 基本清理
    df = df.fillna("")
    
    # 提取必要字段并清理
    patents_df = df[['PubNo-公开号', 'Title-题名', 'Summary-摘要', 'CLC-中图分类号', 'PubTime-公开日期']].copy()
    patents_df.columns = ['patent_id', 'title', 'abstract', 'category', 'publication_date']
    
    # 清理文本字段
    patents_df['title'] = patents_df['title'].apply(clean_text)
    patents_df['abstract'] = patents_df['abstract'].apply(clean_text)
    
    # 创建引用关系（由于示例数据中可能没有引用关系，我们创建一些示例关系）
    relations = []
    patents = patents_df['patent_id'].tolist()
    
    # 为每个专利随机创建1-3个引用关系
    for patent in patents:
        num_citations = random.randint(1, 3)
        possible_citations = [p for p in patents if p != patent]
        if possible_citations:
            citations = random.sample(possible_citations, min(num_citations, len(possible_citations)))
            for cited in citations:
                relations.append({
                    'source_id': patent,
                    'target_id': cited,
                    'relation_type': 'citation'
                })
    
    relations_df = pd.DataFrame(relations)
    
    logger.info(f"Processed {len(patents_df)} patents and {len(relations_df)} relations")
    
    return patents_df, relations_df

def save_processed_data(patents_df: pd.DataFrame, relations_df: pd.DataFrame, output_dir: str):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存专利数据
    patent_file = os.path.join(output_dir, 'Patent.csv')
    patents_df.to_csv(patent_file, index=False)
    logger.info(f"Saved patent data to {patent_file}")
    
    # 保存关系数据
    relations_file = os.path.join(output_dir, 'Patent_with_keys.csv')
    relations_df.to_csv(relations_file, index=False)
    logger.info(f"Saved relations data to {relations_file}")

def main():
    """主函数"""
    try:
        # 创建输出目录
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理小规模测试数据
        input_file = 'data/raw/专利数据/Patent_sm.xlsx'
        patents_df, relations_df = process_patent_data(input_file)
        
        # 保存处理后的数据
        save_processed_data(patents_df, relations_df, output_dir)
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 