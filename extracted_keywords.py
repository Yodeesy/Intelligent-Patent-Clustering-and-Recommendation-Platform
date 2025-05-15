import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('Patent.csv', encoding='utf-8')  # 如果出现编码问题可以尝试 encoding='gbk' 或 'utf-8-sig'

import jieba.analyse

# 提取的关键词将存储在这个列表中
extracted_keywords = []

title_keywords = []  # 用于存储标题的关键词
summary_keywords = []  # 用于存储摘要的关键词
claims_keywords = []  # 用于存储主权项的关键词

# 遍历每一行，合并 Title、Summary 和 Claims
for _, row in df.iterrows():
    # 合并三个字段，使用空格分隔
    combined_text = f"{row['Title-题名']} {row['Summary-摘要']} {row['Claims-主权项']}"
    
    if pd.notnull(combined_text):  # 检查非空
        stopwords = set(['本发明', '一种', '的', '用于','反应','应用','任意','所述','作为','在于','II','所示','特征','以下'])
        #keywords = [word for word in jieba.analyse.extract_tags(combined_text, topK=8) if word not in stopwords]
        #keywords = keywords[:4]  # 取前4个关键词
        # 提取标题的第1个关键词添加到列表中
        title_keyword = [word for word in jieba.analyse.extract_tags(row['Title-题名'], topK=3) if word not in stopwords]
        title_keywords.append(title_keyword[0])  # 将标题的第一个关键词添加到列表中
        # 提取摘要的第1、2个关键词添加到列表中
        summary_keyword = [word for word in jieba.analyse.extract_tags(row['Summary-摘要'], topK=10) if word not in stopwords and word not in title_keyword[0]]
        summary_keyword = summary_keyword[:2]
        summary_keywords.append(';'.join(summary_keyword)) 
        # 提取主权项的第1、2个关键词添加到列表中
        claims_keyword = [word for word in jieba.analyse.extract_tags(row['Claims-主权项'], topK=20) if word not in stopwords and word not in title_keyword[0] and word not in summary_keyword] 
        claims_keyword = claims_keyword[:2]
        claims_keywords.append(';'.join(claims_keyword))

        #extracted_keywords.append(keywords)
        print(f"{_}专利的关键词：{title_keyword[0]} {summary_keyword[:2]} {claims_keyword[:2]}")
    else:
        extracted_keywords.append([])
        print("专利数据为空，无法提取关键词。")
        input()


# 可选择将关键词保存为新的列
df['Title_Keywords-名称关键字'] = title_keywords
df['Summary_Keywords-摘要关键字'] = summary_keywords
df['Claims_Keywords-主权项关键字'] = claims_keywords

# 4. 将新的 DataFrame 保存到新的 CSV 文件
df.to_csv('Patent_with_keys.csv', index=False,encoding='utf-8')
