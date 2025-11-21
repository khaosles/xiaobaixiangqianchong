import os
import json
import re
import asyncio
import time
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pdf_processor import extract_text_from_pdf_base64
from rag_utils import SimpleRAG
from embedding_filter import search_filter_and_cluster
import numpy as np

load_dotenv()

app = FastAPI(title="AI Scientist Challenge Submission")

app.mount("/static", StaticFiles(directory="static"), name="static")

client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

reasoning_client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

embedding_base_url = os.getenv("SCI_EMBEDDING_BASE_URL")
embedding_api_key = os.getenv("SCI_EMBEDDING_API_KEY")
if embedding_base_url and embedding_api_key:
    embedding_client = AsyncOpenAI(
        base_url=embedding_base_url,
        api_key=embedding_api_key
    )
else:
    embedding_client = client


def detect_language(text: str) -> str:
    """
    检测文本的主要语言

    Args:
        text: 输入文本

    Returns:
        'zh' 表示中文，'en' 表示英文
    """
    if not text:
        return 'en'

    # 统计中文字符和英文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = chinese_chars + english_chars

    if total_chars == 0:
        return 'en'

    # 如果完全没有中文字符，判定为英文
    if chinese_chars == 0:
        return 'en'
    
    # 如果完全没有英文字符，判定为中文
    if english_chars == 0:
        return 'zh'

    # 如果中文字符占比超过50%，判定为中文（提高阈值，减少误判）
    if chinese_chars / total_chars > 0.5:
        return 'zh'

    # 默认返回英文
    return 'en'


def format_author_display(author: str) -> tuple:
    """
    格式化作者名称用于显示和引用
    
    规则：
    - 中文作者：返回全名（姓 + 名），不缩写
    - 英文作者：提取姓氏用于显示
    
    Args:
        author: 作者名称字符串
        
    Returns:
        (author_surname, author_display) 元组
        - author_surname: 用于引用的姓氏（中文为全名，英文为姓氏）
        - author_display: 用于显示的格式（中文为全名，英文为姓氏）
    """
    if not author or not isinstance(author, str):
        return 'Unknown', 'Unknown'
    
    author = author.strip()
    if not author:
        return 'Unknown', 'Unknown'
    
    # 检测是否为中文作者（包含中文字符）
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', author))
    
    if has_chinese:
        # 中文作者：使用全名（姓 + 名），不缩写
        # 对于"锦地 王"这样的格式，保持原样或转换为"王锦地"
        # 这里我们保持原样，因为不同来源可能有不同格式
        return author, author
    else:
        # 英文作者处理：提取姓氏
        if ',' in author:
            # 格式：Last, First Middle
            parts = author.split(',')
            surname = parts[0].strip()
            return surname, surname
        else:
            # 格式：First Middle Last
            parts = author.split()
            if len(parts) == 0:
                return author, author
            elif len(parts) == 1:
                return parts[0], parts[0]
            else:
                # 最后一个词是姓
                surname = parts[-1]
                return surname, surname


def format_author_for_gbt7714(author: str) -> str:
    """
    格式化作者名称用于GB/T 7714格式引用
    
    规则：
    - 中文作者：写全名（姓 + 名），不缩写
    - 英文作者：姓全大写或首字母大写 + 名缩写（带点或不带点）
    
    Args:
        author: 作者名称字符串
        
    Returns:
        格式化后的作者名称
    """
    if not author or not isinstance(author, str):
        return 'Unknown'
    
    author = author.strip()
    if not author:
        return 'Unknown'
    
    # 检测是否为中文作者（包含中文字符）
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', author))
    
    if has_chinese:
        # 中文作者：写全名（姓 + 名），不缩写
        return author
    
    # 英文作者处理
    if ',' in author:
        # 格式：Last, First Middle
        parts = author.split(',')
        surname = parts[0].strip()
        given_names = parts[1].strip() if len(parts) > 1 else ''
        
        # 姓：全大写或首字母大写
        if surname:
            surname = surname.upper()  # 全大写
        
        # 名：缩写为首字母（带点或不带点）
        if given_names:
            # 提取名的首字母
            given_initials = []
            for name_part in given_names.split():
                if name_part:
                    given_initials.append(name_part[0].upper())
            if given_initials:
                given_abbr = '. '.join(given_initials) + '.'  # 带点
            else:
                given_abbr = ''
        else:
            given_abbr = ''
        
        if given_abbr:
            return f"{surname} {given_abbr}"
        else:
            return surname
    else:
        # 格式：First Middle Last
        parts = author.split()
        if len(parts) == 0:
            return author
        elif len(parts) == 1:
            # 只有一个词，可能是姓或名
            return parts[0].upper()
        else:
            # 最后一个词是姓，前面的词是名
            surname = parts[-1].upper()
            given_names = parts[:-1]
            
            # 提取名的首字母
            given_initials = []
            for name_part in given_names:
                if name_part:
                    given_initials.append(name_part[0].upper())
            
            if given_initials:
                given_abbr = '. '.join(given_initials) + '.'  # 带点
                return f"{surname} {given_abbr}"
            else:
                return surname


async def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
        
    Returns:
        嵌入向量
    """
    try:
        embedding_model = os.getenv("SCI_EMBEDDING_MODEL", "text-embedding-v4")
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量错误: {e}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        相似度值（0-1之间）
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


async def generate_cluster_name(papers_in_cluster: List[dict], language: str = 'zh') -> str:
    """
    为聚类生成具体的研究方向名称
    
    Args:
        papers_in_cluster: 聚类中的论文列表
        language: 语言 ('zh' 或 'en')
        
    Returns:
        聚类的研究方向名称
    """
    try:
        if not papers_in_cluster:
            return "其他研究方向" if language == 'zh' else "Other Research Directions"
        
        # 提取聚类中论文的标题和摘要
        titles = []
        abstracts = []
        for paper in papers_in_cluster[:7]:  # 只取前5篇论文作为参考
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            if title:
                titles.append(title)
            if abstract:
                abstracts.append(abstract[:150])  # 只取摘要前150字符，减少token数量
        
        if not titles:
            return "其他研究方向" if language == 'zh' else "Other Research Directions"
        
        papers_summary = "\n".join([f"- {title}" for title in titles[:3]])
        if abstracts:
            abstracts_summary = "\n".join([f"- {ab}" for ab in abstracts[:2]])
        else:
            abstracts_summary = ""
        
        if language == 'zh':
            prompt = f"""请根据以下论文的标题和摘要，为这个研究聚类生成一个简洁、准确的研究方向名称。

论文标题：
{papers_summary}

论文摘要（部分）：
{abstracts_summary}

要求：
1. 名称应该简洁明了，3-10个字
2. 准确概括这些论文的共同研究方向
3. 使用学术术语，避免过于宽泛或过于具体
4. 直接输出研究方向名称，不要添加任何解释或标点符号

示例：
- "深度学习模型优化"
- "自然语言处理中的预训练方法"
- "计算机视觉中的目标检测"

研究方向名称："""
        else:
            prompt = f"""Please generate a concise and accurate research direction name for this cluster based on the following paper titles and abstracts.

Paper Titles:
{papers_summary}

Paper Abstracts (partial):
{abstracts_summary}

Requirements:
1. The name should be concise, 3-8 words
2. Accurately summarize the common research direction of these papers
3. Use academic terminology, avoid being too broad or too specific
4. Output the research direction name directly without any explanation or punctuation

Examples:
- "Deep Learning Model Optimization"
- "Pretraining Methods in Natural Language Processing"
- "Object Detection in Computer Vision"

Research Direction Name:"""
        
        response = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        
        cluster_name = response.choices[0].message.content.strip()
        cluster_name = re.sub(r'^[：:、，,。.]+\s*', '', cluster_name)
        cluster_name = re.sub(r'\s*[：:、，,。.]+\s*$', '', cluster_name)
        
        if not cluster_name or len(cluster_name) < 2:
            return "其他研究方向" if language == 'zh' else "Other Research Directions"
        
        return cluster_name
        
    except Exception as e:
        print(f"[generate_cluster_name] 生成聚类名称失败: {e}，使用默认名称")
        return "其他研究方向" if language == 'zh' else "Other Research Directions"


async def extract_research_direction(query: str, language: str = 'en') -> str:
    """
    使用 deepseek-chat 模型提取查询中的主要研究方向
    
    Args:
        query: 用户输入的查询语句
        language: 检测到的语言 ('zh' 或 'en')
        
    Returns:
        提取出的主要研究方向关键词或短语。如果是中文查询，返回格式为 "中文关键词|英文关键词"
    """
    try:
        if language == 'zh':
            prompt = f"""请从以下研究查询中提取最核心的研究方向关键词。

用户查询：{query}

请提取出最核心的研究领域、技术、概念或方法的关键词。必须去除：
- 所有时间描述（如"最新的"、"近期的"、"最新进展"）
- 所有问题表述（如"什么是"、"有哪些"、"如何"）
- 所有修饰词和背景描述（如"最新进展"、"发展趋势"、"应用"）
- 所有动词和形容词

要求：
1. **只提取核心的研究领域、技术、概念或方法的名称**
2. 可以是单个关键词或多个关键词（用逗号分隔）
3. 长度控制在20字以内
4. **必须同时提供中文关键词和对应的英文关键词**
5. 输出格式：中文关键词|英文关键词（用竖线分隔）

输出格式示例：
输入："深度学习的最新进展有哪些？" → 输出："深度学习|deep learning"
输入："transformer模型的最新应用" → 输出："transformer模型|transformer models"
输入："机器学习和神经网络的最新研究" → 输出："机器学习, 神经网络|machine learning, neural networks"

请直接输出关键词（格式：中文关键词|英文关键词）："""
        else:
            prompt = f"""Please extract ONLY the core research direction/keywords from the following research query.

User query: {query}

Extract ONLY the core research field, technology, concept, or method name. You MUST remove:
- All time descriptions (e.g., "latest", "recent", "recent advances", "newest")
- All question phrases (e.g., "what are", "what is", "how to", "what")
- All modifiers and background descriptions (e.g., "latest advances", "developments", "applications", "trends")
- All verbs and adjectives

Requirements:
1. **Extract ONLY the core research field, technology, concept, or method name**
2. Can be a single keyword or multiple keywords (comma-separated)
3. Limit to 50 characters
4. Output keywords directly without any explanation

Example outputs:
Input: "What are the latest advances in transformer models?" → Output: "transformer models"
Input: "recent developments in deep learning" → Output: "deep learning"
Input: "new research on neural networks and machine learning" → Output: "neural networks, machine learning"
Input: "最新进展在transformer模型" → Output: "transformer models"

Please output keywords directly:"""
        
        response = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1  # 使用较低温度确保输出更稳定
        )
        
        extracted_keywords = response.choices[0].message.content.strip()
        
        if not extracted_keywords or len(extracted_keywords) < 3:
            return query
        
        if language == 'zh' and '|' in extracted_keywords:
            return extracted_keywords
        print("extracted_keywords: ", extracted_keywords)
        return extracted_keywords
        
    except Exception as e:
        return query


@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses standard LLM model
    
    Request body:
    {
        "query": "What are the latest advances in transformer models?"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        language = detect_language(query)

        async def generate():
            start_time = time.time()
            first_token_time = None
            try:
                papers_context = ""
                papers_references = []
                try:
                    research_keywords = await extract_research_direction(query, language)
                    
                    search_queries = []
                    if language == 'zh' and '|' in research_keywords:
                        parts = research_keywords.split('|', 1)
                        chinese_keywords = parts[0].strip()
                        english_keywords = parts[1].strip() if len(parts) > 1 else ''
                        if chinese_keywords:
                            search_queries.append(chinese_keywords)
                        if english_keywords:
                            search_queries.append(english_keywords)
                    else:
                        search_queries = [research_keywords]
                    
                    all_filtered_papers = []
                    all_clusters = {}
                    
                    for search_query in search_queries:
                        search_result = await search_filter_and_cluster(
                            query=search_query,
                            max_results_per_source=15,  # 每个数据源最多10篇
                            similarity_threshold=0.27,
                            top_k=45,  # 最多保留36篇高相似度论文
                            n_clusters=None,  # 自动确定聚类数量
                            min_cluster_size=2
                        )
                        
                        filtered_papers = search_result.get('filtered_papers', [])
                        clusters = search_result.get('clusters', {})
                        
                        paper_ids = {paper.get('title', '') for paper in all_filtered_papers}
                        for paper in filtered_papers:
                            if paper.get('title', '') not in paper_ids:
                                all_filtered_papers.append(paper)
                                paper_ids.add(paper.get('title', ''))
                        
                        max_cluster_id = max([int(k) for k in all_clusters.keys()], default=-1)
                        for cluster_id, papers_in_cluster in clusters.items():
                            new_cluster_id = str(max_cluster_id + 1 + int(cluster_id))
                            all_clusters[new_cluster_id] = papers_in_cluster
                    
                    clusters = all_clusters
                    filtered_papers = all_filtered_papers
                    
                    if clusters:
                        cluster_sections = []
                        cluster_names = {}  # 存储聚类名称
                        global_paper_idx = 1
                        
                        sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]))
                        cluster_tasks = [
                            generate_cluster_name(papers_in_cluster, language)
                            for _, papers_in_cluster in sorted_clusters
                        ]
                        cluster_name_results = await asyncio.gather(*cluster_tasks)
                        for (cluster_id, _), cluster_name in zip(sorted_clusters, cluster_name_results):
                            cluster_names[cluster_id] = cluster_name
                        
                        for cluster_id, papers_in_cluster in sorted(clusters.items(), key=lambda x: int(x[0])):
                            cluster_papers_list = []
                            cluster_name = cluster_names.get(cluster_id, f"聚类 {cluster_id}" if language == 'zh' else f"Cluster {cluster_id}")
                            
                            for paper in papers_in_cluster:
                                authors = paper.get('authors', [])[:3] if paper.get('authors') else []
                                year = paper.get('year')
                                year_str = str(year) if year else ''
                                
                                author_surname = ''
                                author_display = ''
                                if authors:
                                    first_author = authors[0]
                                    if first_author and isinstance(first_author, str):
                                        author_surname, first_display = format_author_display(first_author)
                                    else:
                                        author_surname = 'Unknown'
                                        first_display = 'Unknown'
                                    
                                    if len(authors) == 1:
                                        author_display = first_display
                                    elif len(authors) == 2:
                                        second_author = authors[1]
                                        if second_author and isinstance(second_author, str):
                                            _, second_display = format_author_display(second_author)
                                        else:
                                            second_display = 'Unknown'
                                        author_display = f"{first_display} and {second_display}"
                                    else:
                                        author_display = f"{first_display} et al."
                                else:
                                    author_surname = 'Unknown'
                                    author_display = 'Unknown'
                                
                                title = paper.get('title', '未知' if language == 'zh' else 'Unknown')
                                abstract = paper.get('abstract', '')[:444] if paper.get('abstract') else ''
                                url = paper.get('url', '')
                                doi = paper.get('doi', '')
                                citation_url = f"https://doi.org/{doi}" if doi else url
                                
                                if language == 'zh':
                                    paper_info = f"{global_paper_idx}. 标题: {title}\n   作者: {author_display}"
                                    if year_str:
                                        paper_info += f"\n   年份: {year_str}"
                                    paper_info += f"\n   摘要: {abstract}"
                                    if citation_url:
                                        paper_info += f"\n   URL: {citation_url}"
                                else:
                                    paper_info = f"{global_paper_idx}. Title: {title}\n   Authors: {author_display}"
                                    if year_str:
                                        paper_info += f"\n   Year: {year_str}"
                                    paper_info += f"\n   Abstract: {abstract}"
                                    if citation_url:
                                        paper_info += f"\n   URL: {citation_url}"
                                
                                cluster_papers_list.append(paper_info)
                                
                                ref_info = {
                                    'author_surname': author_surname,
                                    'author_display': author_display,
                                    'title': title,
                                    'authors': authors,
                                    'published': year_str,
                                    'year': year_str,
                                    'url': url,
                                    'doi': doi,
                                    'citation_url': citation_url,
                                    'abstract': abstract,
                                    'multiple_authors': len(authors) > 1
                                }
                                papers_references.append(ref_info)
                                global_paper_idx += 1
                            
                            if cluster_papers_list:
                                if language == 'zh':
                                    cluster_section = f"**{cluster_name}** ({len(papers_in_cluster)} 篇论文):\n\n" + "\n\n".join(cluster_papers_list)
                                else:
                                    cluster_section = f"**{cluster_name}** ({len(papers_in_cluster)} papers):\n\n" + "\n\n".join(cluster_papers_list)
                                cluster_sections.append(cluster_section)
                        
                        if cluster_sections:
                            if language == 'zh':
                                papers_context = "\n\n提供的文献（按聚类组织）:\n\n" + "\n\n---\n\n".join(cluster_sections)
                            else:
                                papers_context = "\n\n**Provided Papers (Organized by Clusters):**\n\n" + "\n\n---\n\n".join(cluster_sections)
                    else:
                        papers_list = []
                        for idx, paper in enumerate(filtered_papers, 1):
                            authors = paper.get('authors', [])[:3] if paper.get('authors') else []
                            year = paper.get('year')
                            year_str = str(year) if year else ''
                            
                            author_surname = ''
                            author_display = ''
                            if authors:
                                first_author = authors[0]
                                if first_author and isinstance(first_author, str):
                                    author_surname, first_display = format_author_display(first_author)
                                else:
                                    author_surname = 'Unknown'
                                    first_display = 'Unknown'
                                
                                if len(authors) == 1:
                                    author_display = first_display
                                elif len(authors) == 2:
                                    second_author = authors[1]
                                    if second_author and isinstance(second_author, str):
                                        _, second_display = format_author_display(second_author)
                                    else:
                                        second_display = 'Unknown'
                                    author_display = f"{first_display} and {second_display}"
                                else:
                                    author_display = f"{first_display} et al."
                            else:
                                author_surname = 'Unknown'
                                author_display = 'Unknown'
                            
                            title = paper.get('title', '未知' if language == 'zh' else 'Unknown')
                            abstract = paper.get('abstract', '')[:444] if paper.get('abstract') else ''
                            url = paper.get('url', '')
                            doi = paper.get('doi', '')
                            citation_url = f"https://doi.org/{doi}" if doi else url
                            
                            if language == 'zh':
                                paper_info = f"{idx}. 标题: {title}\n   作者: {author_display}"
                                if year_str:
                                    paper_info += f"\n   年份: {year_str}"
                                paper_info += f"\n   摘要: {abstract}"
                                if citation_url:
                                    paper_info += f"\n   URL: {citation_url}"
                            else:
                                paper_info = f"{idx}. Title: {title}\n   Authors: {author_display}"
                                if year_str:
                                    paper_info += f"\n   Year: {year_str}"
                                paper_info += f"\n   Abstract: {abstract}"
                                if citation_url:
                                    paper_info += f"\n   URL: {citation_url}"
                            
                            papers_list.append(paper_info)
                            
                            ref_info = {
                                'author_surname': author_surname,
                                'author_display': author_display,
                                'title': title,
                                'authors': authors,
                                'published': year_str,
                                'year': year_str,
                                'url': url,
                                'doi': doi,
                                'citation_url': citation_url,
                                'abstract': abstract,
                                'multiple_authors': len(authors) > 1
                            }
                            papers_references.append(ref_info)
                        
                        if papers_list:
                            if language == 'zh':
                                papers_context = "\n\n提供的文献:\n\n" + "\n\n".join(papers_list)
                            else:
                                papers_context = "\n\n**Provided Papers:**\n\n" + "\n\n".join(papers_list)
                except Exception as e:
                    print(f"[literature_review] 文献搜索警告: {e}，继续生成综述")
                    import traceback
                    traceback.print_exc()

                if language == 'zh':
                    if papers_context:
                        citation_rules = """
**引用规则（APA格式）:**

- **必须尽可能使用提供的全部文献。请仔细阅读所有提供的文献，并在综述中充分引用它们。**
- 只能引用下面列出的文献。不要编造标题、作者、年份或研究成果。
- **严禁引入没有给定的文献。所有引用必须来自提供的文献列表。**
  
  **正文中引用格式（APA格式）：**
  引用时可以使用两种格式：
  1. 括号内引用格式：(作者, 年份) 或 (作者等, 年份)
  2. 作者在前格式：作者 (年份) 或 作者等 (年份)
  
  **作者引用规则（APA格式）：**
  - **单个作者**：(作者, 年份) 或 作者 (年份)
  - **两个作者**：使用"和"或"&"连接，如 (作者1和作者2, 年份) 或 作者1和作者2 (年份)
  - **三个或更多作者（正文引用）**：首次引用时列出所有作者，如 (作者1, 作者2, & 作者3, 年份)；后续引用或三个以上作者用"等"，如 (作者1等, 年份)
  - **中文作者**：使用中文全名（姓 + 名），如 (张三, 2024) 或 张三 (2024)
  - **英文作者**：使用姓氏，如 (Chen, 2024) 或 Chen (2024)
  
  **示例：**
  - 单个作者："根据研究 (张三, 2024)..." 或 "张三 (2024) 研究发现..."
  - 两个作者："研究表明 (张三和李四, 2024)..." 或 "张三和李四 (2024) 提出..."
  - 多个作者："研究发现 (张三, 李四, & 王五, 2024)..." 或 "张三等 (2024) 指出..."
  - 英文作者："According to Chen (2024)..." 或 "(Chen et al., 2024)"

- 在综述末尾，必须包含一个 **参考文献** 部分，按字母顺序（或出现顺序）列出所有引用的文献，严格使用APA格式。
  
  **重要：参考文献必须包含所有在正文中被引用的文献，包括中文文献和英文文献，不能遗漏任何一篇。**
  
  **参考文献格式（APA格式）：**
  
  APA格式示例（英文期刊文章）：
  Chen, X., Wang, Y., & Zhang, Z. (2024). Diffusion language models are versatile few-shot learners. *Journal Name*, *15*(3), 123-145. https://doi.org/10.xxxx/xxxx
  
  APA格式示例（英文会议论文）：
  Chen, X., & Wang, Y. (2023, July). Latent diffusion for text generation. In *Proceedings of the Conference Name* (pp. 123-134). Publisher. https://aclanthology.org/2023.xxx.pdf
  
  APA格式示例（中文期刊文章）：
  张三, 李四, & 王五. (2024). 深度学习在自然语言处理中的应用. *计算机学报*, *47*(5), 123-145. https://doi.org/10.xxxx/xxxx

  **APA格式详细规则：**
  - **作者格式**：
    * 英文作者：Last, F. M. (姓, 名首字母大写，加点)，如 Smith, J. K.
    * 中文作者：姓 名 (中文全名)，如 张三
    * 多个作者：用逗号分隔，最后一个作者前用 & 连接，如 Smith, A., Jones, B., & Wang, C. (2024)
    * 参考文献中：超过20个作者时才在列出前19个后用 et al.，否则列出所有作者
  - **年份**：放在作者后面的括号中，如 (2024)
  - **标题格式**：
    * 文章标题：只有首字母和专有名词大写，标题末尾不加句号
    * 期刊名称：斜体，每个主要单词首字母大写
    * **必须包含完整的论文标题，不能省略或缩写**
  - **期刊格式**：作者. (年份). 文章标题. *期刊名称*, *卷号*(期号), 页码. DOI或URL
  - **会议格式**：作者. (年份, 月份). 论文标题. In *会议名称* (pp. 页码). 出版社. URL
  - **DOI格式**：https://doi.org/10.xxxx/xxxx 或 https://doi.org/xx.xxx/yyyy
  - **重要**：每条参考文献必须包含：作者（完整格式）、年份、标题（完整准确）、期刊名/会议名（斜体）、卷期页码或会议信息、以及DOI或URL（如果有）。标题是必需组成部分，不能省略。

- **必须确保参考文献列表包含所有在正文中引用的文献，无论是中文文献还是英文文献，都不能遗漏。**
- 如果文献在正文中没有被引用，不要将其包含在参考文献中。
- 所有观点必须严格基于提供的摘要。
- **如果给定的文献列表为空，则跳过参考文献部分，不要生成任何参考文献。**
"""
                    else:
                        citation_rules = ""

                    prompt = f"""**重要：你必须使用中文撰写这篇文献综述。无论提供的文献是中文还是英文，你都必须用中文撰写综述内容。**

请对以下研究主题进行全面的文献综述：

{query}

{papers_context}

{citation_rules}

**输出要求：**
- **必须在输出的第一行开始生成文章标题，使用格式 `# **[文章标题]**`**
- **标题必须加粗，完整显示在输出内容的最开始位置**
- **之后紧接着输出摘要部分，然后是其他章节**

**重要：在开始写作之前，必须先完成以下综合分析步骤：**

**第一步：全面阅读和综合分析所有文献（必须完成此步骤后才能开始写作）**
1. **全面阅读**：仔细阅读所有提供的文献摘要，理解每篇文献的核心内容、研究方法、主要发现和结论
2. **识别主题**：识别所有文献中涉及的主要研究主题、概念、理论框架、方法论和技术
3. **找出共同点**：分析所有文献之间的共同观点、共同方法、共同发现和共同趋势
4. **找出差异点**：识别不同文献之间的不同观点、不同方法、不同发现和争议点
5. **综合归纳**：基于所有文献，形成综合性的理解，包括：
   - 该领域的核心研究方向和主题
   - 主要的研究方法和理论框架
   - 当前研究的主要共识和争议
   - 研究的发展趋势和未来方向
6. **组织框架**：根据综合分析结果，确定文献综述的组织框架和逻辑结构

**第二步：基于综合分析结果生成文献综述**
只有在完成第一步的全面综合分析后，才能开始写作。写作时：
- 必须基于第一步的综合分析结果
- 每个观点都必须有多个文献支撑
- 必须体现文献之间的共同点和差异点
- 必须形成综合性的论述，而不是简单罗列单篇文献的观点

请提供一份结构化的文献综述，使用Markdown格式，严格按照以下结构组织内容：

**重要提示：必须尽可能使用提供的全部文献。在撰写每个部分时，必须在适当的地方引用提供的文献。每个主要观点、研究发现、技术方法都应该有相应的文献引用支持。请确保在综述中引用了大部分或全部提供的文献。**

**关于摘要的使用：**
- **请仔细阅读每篇论文的摘要，摘要包含了论文的核心内容、研究方法、主要发现和结论**
- **在撰写综述时，要充分基于摘要中的信息，提取每篇论文的关键观点、研究方法、实验结果和主要贡献**
- **在引用文献时，要准确反映摘要中描述的研究内容，不要编造或推测摘要中没有的信息**
- **对于不同聚类中的论文，要分析其摘要中的共同主题和差异，在综述中体现这些研究方向的联系和区别**

**标题要求（必须遵守）：**
- **文章标题必须在输出内容的第一行开始生成**
- **文章标题格式：使用 `# **[文章标题]**` 格式，标题必须加粗显示**
- **直接输出标题，不要写"## 1. 标题（Title）"这样的格式**
- **标题必须准确反映综述主题，可包含"综述""进展""现状与展望"等关键词**
- **标题应简洁明了，突出研究领域，力求做到简短、鲜明、确切，能够用最短的文字概括全文精要**

**章节标题格式要求：**
- 所有一级标题（如 `## 1. 引言`、`## 2. 基本概念和理论背景`、`## 3. [主题名称]`、`## [N]. 讨论`、`## [N+1]. 结论`）必须加粗显示
- 使用格式：`## **1. 引言**` 或 `## **2. 基本概念和理论背景**`
- 二级标题（如 `### 2.1 核心概念X`）不需要加粗，使用普通格式：`### 2.1 核心概念X`

## 摘要
摘要是对整篇文献综述简明扼要的总结，具有高度抽象性和概括性。综述类论文的摘要一般包含：
- **写作目的**：说明本综述的写作目的和意义
- **资料来源**：说明文献来源（如 OpenAlex, arXiv 等数据库）
- **资料提取和整合的过程和结果**：简要说明文献检索、筛选和整合的方法，以及主要发现
- **论文的结论**：总结该领域的主要研究方向和核心问题，概述当前研究的主要进展和重要发现，简要说明该领域的发展趋势和未来研究方向
- 摘要应简洁明了，通常200-300字左右
- 在摘要末尾列出3-6个关键词（Keywords），关键词包括论文所涉及的主要概念

## 1. 引言
引言部分要开门见山、简单精炼，字数一般为两三百字，目的是为了让读者迅速进入正题并产生继续阅读的兴趣。引言一般包括：
- 阐明研究领域的背景与重要性
- 简要介绍该领域的历史发展和当前研究状况
- 明确综述的研究问题或目标，总结该领域的核心问题和研究动机
- 简要说明该领域的发展趋势和未来研究方向
- 界定综述的范围（时间、地域、学科、方法等），说明综述的组织结构

对于系统或半系统综述，需要详细说明文献检索和筛选的方法，包括：
- 说明使用的数据库（如 OpenAlex, arXiv 等）
- 说明检索关键词、检索式、检索时间范围等
- 明确文献的筛选标准（纳入与排除标准）
- 简要说明文献筛选的过程
- 如适用，说明文献质量评估的方法

在引言部分的最后，应综合讨论本综述将要涵盖的主要文献和研究，包括：
- 明确列出本综述将要讨论的主要文献和研究
- 简要说明文献是如何分类组织的（如按主题、方法论、时间等）
- 说明文献的时间跨度、研究类型、主要研究问题等
- 如果文献按聚类组织，应说明每个聚类代表的研究方向以及每个聚类中包含的主要研究

## 2. 基本概念和理论背景（Basic Concepts and Theoretical Background）
本章节应根据提供的文献中涉及的主要概念和理论基础，分小节进行详细介绍。每个小节应聚焦一个核心概念或理论框架。

**重要：全面综合原则（本节特别强调）**
- **在撰写每个小节之前，必须先全面阅读所有涉及该概念或理论的文献**
- **不要只看到一篇文章就介绍一个概念，必须综合分析多篇相关文献**
- **找出这些文献中的共同观点、不同观点和综合观点，然后进行综合归纳**
- **每个概念的介绍必须基于多篇文献的综合分析，确保观点的全面性和代表性**
- **避免片面性：不要仅基于单一文献就介绍一个概念或理论，要引用多篇文献来支撑综合观点**

**组织方式：**
- 仔细阅读提供的文献，识别文章中涉及的所有主要概念、理论框架、方法论、技术原理等
- 根据识别出的概念和理论，分为多个小节（如2.1、2.2、2.3等）
- 每个小节聚焦一个核心概念或理论背景
- 使用清晰的子标题，直接使用概念或理论的名称，例如：
  * "2.1 核心概念X"
  * "2.2 理论框架Y"
  * "2.3 方法论Z"
  * "2.4 技术W"
  * 或根据文献中实际涉及的主要概念和理论名称创建小节标题（只使用概念或理论的名字，不要添加"定义"、"发展"、"原理"等描述性词语）

**每个小节应包含（必须基于多篇文献的综合分析）：**
- **概念/理论的定义**：清晰定义该概念或理论，必须综合多篇文献的定义，找出共同点，如果有不同观点也要说明
- **历史发展**：介绍该概念或理论的提出及发展过程，必须综合多篇相关文献，全面梳理发展脉络，引用多篇文献
- **核心内容**：阐述该概念或理论的核心内容、关键要素、基本原理，必须基于多篇文献的综合分析，找出共同的核心观点
- **理论关系**：说明与其他概念或理论之间的关系和演变，必须综合分析相关文献的观点
- **应用领域**：介绍该概念或理论在该研究领域的应用，必须引用多篇文献，全面介绍应用情况
- **主要观点**：总结文献中关于该概念或理论的主要观点，必须综合分析所有相关文献，找出共同观点、不同观点和综合观点
- **必须引用多篇相关文献支持每个概念或理论的介绍，不能只引用一篇文章**

**主体部分的组织说明：**
主体部分是文献综述的核心，应按照主题或方法论进行分类组织。根据文献聚类结果或研究主题，为每个主题创建独立的章节（从第3章开始编号）。每个章节应使用清晰、信息丰富的标题，例如：
- "X的早期概念模型"（历史演变类）
- "Y的实验研究"（研究方法类）
- "关于Z的定性视角"（方法论类）
- "测量方法的方法论挑战"（方法学类）
- "跨文化比较和情境效应"（比较研究类）
- 或根据聚类标题中给出的研究方向名称创建标题

**每个主题章节应包含：**
- **主要观点**：提出本节想要阐述的主要观点和发现
- **对整体论点的贡献**：说明本节如何有助于整体论点或目标
- 对相关研究进行归纳、比较与批判性分析
- **严禁简单罗列文献**：不要逐篇介绍该主题中的文献，不要写成"文献A做了什么，文献B做了什么"的形式
- **必须总结分析大家的做法**：综合分析该主题中多篇文献的研究方法、研究思路、研究结果，总结出：
  * 该主题下共同的研究方法和做法
  * 不同的研究方法和做法及其优缺点
  * 研究方法和做法的发展趋势
  * 当前该主题研究的主要方向和热点
- 引用代表性文献并指出其贡献与局限
- 详细讨论该聚类或主题中的相关研究，阐述研究发现、方法和贡献，但要基于综合分析，而不是简单罗列

**注意：** 如果文献按聚类组织，每个聚类代表一个研究主题，应为每个聚类创建一个独立的章节（从第3章开始，依次编号为3、4、5...）。请使用聚类标题中给出的研究方向名称作为章节标题，不要使用"聚类 0"、"聚类 1"等编号。每个章节应：
1. 讨论该聚类中的所有相关研究
2. 提出该研究方向的主要观点和发现
3. 说明该研究方向如何有助于整体综述的目标
4. 如果该主题涉及历史演变，采用纵向对比和分析法，分析问题或理论的提出及发展的过程和脉络
5. 如果该主题涉及当前研究状况，采用横向分析，对比当下或近期的文献，揭示尚未解决的问题和热点问题
6. 如果该主题涉及未来预测，基于历史和当前文献预测未来可能的发展方向

**可选内容：**
- 如果有助于读者理解，可以在某个主题章节中总结该领域的历史发展
- 可以考虑展示理论或变量之间关系的概念框架
- 可以考虑创建表格比较主要研究的关键特征

## [N]. 讨论
讨论部分应对主体部分的内容进行综合分析、解释和评价：
- **综合主要发现**：整合主体部分各小节的主要发现
- **解释和意义**：解释研究发现的意义，分析不同研究之间的联系和差异
- **理论贡献**：讨论研究对理论发展的贡献
- **方法学评价**：评价现有研究的方法学优势和局限性
- **研究空白**：指出当前研究的不足和空白
- **争议和分歧**：分析不同学者或研究之间的争议和分歧

## [N+1]. 结论
- **综合全文核心发现**：总结综述的主要发现和贡献
- **强调该领域知识的整体进展**：概述该领域的整体发展状况
- **重申综述的学术或实践价值**：说明综述的意义和价值

## 参考文献
- 按**APA第7版格式**规范列出所有在正文中实际引用的文献
- 排序方式：**按作者姓氏字母顺序排列**
- 确保涵盖经典文献与近3–5年最新成果（若所提供文献包含此类作品）
- **仅包含正文中明确引用的文献**，不得列入未引用的参考项
- 每条参考文献必须包含：作者、年份、标题、来源（期刊名/会议名/arXiv ID等）

{citation_rules}

写作要求：

**文献综述写作的重要原则（必须严格遵守）：**

**核心原则：先全面综合分析，再综合生成**
- **第一步：全面综合分析所有文献（写作前必须完成）**
  - 必须仔细阅读所有提供的文献摘要
  - 识别所有文献中的主要主题、概念、理论和方法
  - 找出所有文献之间的共同点、差异点和综合观点
  - 形成对该领域的全面、综合性的理解
  - 确定综述的组织框架和逻辑结构
  
- **第二步：基于综合分析结果生成综述（完成第一步后才能开始）**
  - 所有观点必须基于第一步的综合分析结果
  - 每个观点都必须有多个文献支撑，不能只基于单一文献
  - 必须体现文献之间的共同点和差异点
  - **严禁简单罗列文献**：不要逐篇介绍文献内容，不要写成"文献A做了什么，文献B做了什么"这样的罗列形式
  - **必须总结分析大家的做法**：要综合分析多篇文献的研究方法、研究思路、研究结果，总结出共同的做法、不同的做法、优缺点和趋势
  - 必须形成综合性的论述，而不是简单罗列单篇文献的观点
  - 必须确保观点的全面性和代表性

**特别强调：**
- **在第二节（基本概念和理论背景）中，这一点尤为重要**。在介绍每个概念或理论时，必须：
  1. 首先全面阅读所有涉及该概念或理论的文献
  2. 找出这些文献中的共同观点、不同观点和综合观点
  3. 综合分析后再进行表达，确保观点的全面性和准确性
  4. 避免只引用一篇文章就介绍一个概念，应该引用多篇文献来支撑综合观点
- **在所有章节中，都必须遵循"先综合分析，再综合生成"的原则**

**格式和结构要求：**
- **必须使用中文撰写整篇综述，包括所有章节、引用和参考文献部分**
- **必须在输出内容的第一行开始生成标题，使用 `# **[文章标题]**` 格式，标题必须加粗显示**
- **所有一级标题（章节标题）必须加粗显示，格式为 `## **1. 引言**`、`## **2. 基本概念和理论背景**` 等**
- 严格按照上述结构组织内容：标题（不编号）、摘要（不编号）、1章节（引言与方法/搜索策略）、2章节（基本概念和理论背景）、3-N章节（根据聚类数量动态创建的主题章节，每个聚类一个独立章节）、讨论章节、结论章节、参考文献（不编号），不要遗漏任何部分
- **主题章节的组织**：根据文献聚类结果，为每个聚类创建一个独立的章节（从第3章开始编号）。如果文献没有聚类，可以根据研究主题、方法论或时间维度组织主题章节
- **主题章节的标题**：使用清晰、信息丰富的标题，每个标题下明确讨论的研究、主要观点和对整体论点的贡献
- **逻辑进展**：确保从背景和定义，经过证据和分析，最终到解释和意义的逻辑进展清晰
- 保持逻辑清晰，各部分之间衔接自然
- 不要输出多余的空白行，保持格式紧凑

**内容质量要求：**
- 确保综述全面、准确、有深度，体现对领域的深入理解
- 写作时要写得全面一点，内容丰富一些，深入分析每个主题，提供详细的论述和解释
- **严禁简单罗列文献**：
  - 不要逐篇介绍文献，不要写成"张三等(2020)研究了X，李四等(2021)研究了Y"这样的简单罗列
  - 不要一篇一篇地介绍每篇文献做了什么，而是要综合分析多篇文献的做法
  - 要总结分析大家的做法：综合分析多篇文献的研究方法、研究思路、研究结果，总结出：
    * 共同的研究方法和做法
    * 不同的研究方法和做法
    * 各种方法的优缺点
    * 研究方法和做法的发展趋势
    * 当前研究的主要方向和热点
- **必须严格遵循引用规则，在每个部分都要在适当的地方引用文献，不能遗漏引用**
- **每个主要观点、研究发现、技术方法都必须有相应的文献引用支持**
- **要充分利用每篇论文的摘要信息，在综述中准确反映摘要中描述的研究内容、方法和发现**
- **对于按聚类组织的文献，要在综述中体现不同聚类的研究主题差异和联系**
- 在适当的地方使用作者年份格式的文献引用，格式为：(作者, 年份) 或 作者 (年份)
- **重要：对于中文作者，必须使用全名（姓 + 名），不能缩写；对于英文作者，可以使用姓氏或姓氏+等**"""
                else:
                    if papers_context:
                        citation_rules = """
**Rules (APA format):**

- **You MUST use as many of the provided papers as possible. Please carefully read all provided papers and cite them extensively in your review.**
- Cite only papers listed below. DO NOT invent titles, authors, years, or findings.
- **DO NOT introduce any papers that are not provided. All citations must come from the provided paper list.**
  
  **In-text citation format (APA style):**
  You can use two formats when citing:
  1. Parenthetical citation: (Author, Year) or (Author et al., Year)
  2. Narrative citation: Author (Year) or Author et al. (Year)
  
  **Author citation rules (APA style):**
  - **Single author**: (Author, Year) or Author (Year)
  - **Two authors**: Use "&" to connect, e.g., (Author1 & Author2, Year) or Author1 & Author2 (Year)
  - **Three or more authors (in-text citation)**: First citation list all authors, e.g., (Author1, Author2, & Author3, Year); subsequent citations or three+ authors use "et al.", e.g., (Author1 et al., Year)
  - **Chinese authors**: Use full Chinese name (surname + given name), e.g., (Zhang, 2024) or Zhang (2024)
  - **English authors**: Use surname only, e.g., (Chen, 2024) or Chen (2024)
  
  **Examples:**
  - Single author: "According to research (Zhang, 2024)..." or "Zhang (2024) found..."
  - Two authors: "Research shows (Zhang & Li, 2024)..." or "Zhang & Li (2024) proposed..."
  - Multiple authors: "Research found (Zhang, Li, & Wang, 2024)..." or "Zhang et al. (2024) noted..."

- At the end, you MUST include a **References** section listing all cited papers in **APA 7th edition format** that are actually cited in the text.

  **IMPORTANT: The References section MUST include ALL papers cited in the text, including both Chinese and English papers. Do NOT omit any cited paper.**
  
  **Sorting method**: **Arrange by author surname in alphabetical order**

  APA format example (English journal):
  Chen, X., Wang, Y., & Zhang, Z. (2024). Diffusion language models are versatile few-shot learners. Journal Name, 15(3), 123-145. https://doi.org/10.xxxx/xxxx
  
  APA format example (English conference):
  Chen, X., & Wang, Y. (2023, July). Latent diffusion for text generation. In Proceedings of the Conference Name (pp. 123-134). Publisher. https://aclanthology.org/2023.xxx.pdf
  
  APA format example (Chinese journal):
  Zhang, S., Li, S., & Wang, W. (2024). 深度学习在自然语言处理中的应用 [Deep learning applications in natural language processing]. 计算机学报 [Journal of Computers], 47(5), 123-145. https://doi.org/10.xxxx/xxxx

  APA format notes:
  - Author format:
    * English authors: Last, F. M. (surname, first initial), multiple authors connected with &
    * Chinese authors: Surname Given Name (full name), multiple authors connected with &
  - Year: Placed in parentheses after authors
  - Title format:
    * Article title: Only first letter and proper nouns capitalized, no period at end
    * Journal name: Italicized, each major word capitalized
    * **MUST include the complete paper title, do NOT omit or abbreviate**
  - Journal format: Journal Name, Volume(Issue), pages. DOI or URL
  - Conference format: In Conference Name (pp. pages). Publisher. URL
  - Multiple authors: Use & to connect, not "and" or comma, e.g., Smith, A., & Jones, B. (2024)
  - Use et al. only when there are more than 20 authors, otherwise list all authors
  - DOI format: https://doi.org/10.xxxx/xxxx or https://doi.org/xx.xxx/yyyy
  - **IMPORTANT: Each reference MUST include: Authors (complete format), Year, Title (complete and accurate), Journal/Conference Name (italicized), Volume/issue and pages or conference info, and DOI/URL if available. The title is a required component and must NOT be omitted.**

- **You MUST ensure that the References section includes ALL papers cited in the text, whether Chinese or English papers. Do NOT omit any cited paper.**
- If a paper is not cited in the text, do NOT include it in References.
- Base all claims strictly on the abstracts provided.
- **If the provided papers list is empty, skip the References section entirely and do NOT generate any references.**
"""
                    else:
                        citation_rules = ""

                    prompt = f"""**IMPORTANT: You MUST write this literature review in English. Regardless of whether the provided papers are in Chinese or English, you MUST write the review content in English.**

Conduct a comprehensive literature review on the following topic:

{query}

{papers_context}

{citation_rules}

**Output Requirements:**
- **MUST generate the article title at the very first line of your output, using format `# **[Article Title]**`**
- **The title MUST be bold and displayed completely at the beginning of the output**
- **Followed immediately by the Abstract section, then other sections**

**IMPORTANT: Before starting to write, you MUST complete the following comprehensive analysis steps:**

**Step 1: Comprehensive Reading and Analysis of All Papers (MUST complete this step before writing)**
1. **Comprehensive Reading**: Carefully read all provided paper abstracts, understand the core content, research methods, main findings, and conclusions of each paper
2. **Identify Themes**: Identify all major research themes, concepts, theoretical frameworks, methodologies, and technologies involved in all papers
3. **Find Common Points**: Analyze common viewpoints, common methods, common findings, and common trends among all papers
4. **Find Differences**: Identify different viewpoints, different methods, different findings, and controversies among different papers
5. **Synthesize and Summarize**: Based on all papers, form a comprehensive understanding, including:
   - Core research directions and themes in the field
   - Main research methods and theoretical frameworks
   - Main consensus and controversies in current research
   - Research development trends and future directions
6. **Organize Framework**: Based on the comprehensive analysis results, determine the organizational framework and logical structure of the literature review

**Step 2: Generate Literature Review Based on Comprehensive Analysis Results**
Only after completing Step 1's comprehensive analysis can you start writing. When writing:
- MUST be based on the comprehensive analysis results from Step 1
- Every viewpoint must be supported by multiple papers
- MUST reflect common points and differences among papers
- MUST form synthesized discussions, not simply list viewpoints from individual papers

Please provide a structured literature review in Markdown format, strictly following the structure below:

**IMPORTANT: You MUST use as many of the provided papers as possible. When writing each section, you MUST cite the provided papers at appropriate places. Every major point, research finding, or technical method should be supported by relevant citations. Please ensure that you cite most or all of the provided papers in your review.**

**About Using Abstracts:**
- **Please carefully read the abstract of each paper, as abstracts contain the core content, research methods, main findings, and conclusions of the papers**
- **When writing the review, fully base your content on the information in the abstracts, extracting key viewpoints, research methods, experimental results, and main contributions from each abstract**
- **When citing papers, accurately reflect the research content described in the abstracts, and do not invent or speculate on information not present in the abstracts**
- **For papers in different clusters, analyze the common themes and differences in their abstracts, and reflect the connections and distinctions between these research directions in the review**

**Title Requirements (MUST follow):**
- **The article title MUST be generated at the very first line of the output**
- **Article title format: Use `# **[Article Title]**` format, the title MUST be bold**
- **Output the title directly, do NOT write "## 1. Title" format**
- **The title MUST accurately reflect the review topic, may include keywords such as "Review", "Advances", "Current Status and Prospects"**
- **The title should be concise and highlight the research field, strive to be brief, clear, and precise, using the shortest text to summarize the essence of the entire review**

**Section Heading Format Requirements:**
- All first-level headings (e.g., `## 1. Introduction`, `## 2. Basic Concepts and Theoretical Background`, `## 3. [Theme Name]`, `## [N]. Discussion`, `## [N+1]. Conclusion`) MUST be bold
- Use format: `## **1. Introduction**` or `## **2. Basic Concepts and Theoretical Background**`
- Second-level headings (e.g., `### 2.1 Core Concept X`) should NOT be bold, use normal format: `### 2.1 Core Concept X`

## Abstract
The abstract is a concise summary of the entire literature review with high abstraction and generalization. For review papers, the abstract generally includes:
- **Writing purpose**: Explain the purpose and significance of this review
- **Data sources**: Describe the literature sources (e.g., OpenAlex, arXiv databases)
- **Data extraction and integration process and results**: Briefly explain the methods of literature search, screening, and integration, as well as main findings
- **Conclusions**: Summarize the main research directions and core problems in the field, outline the major advances and important findings in current research, briefly describe the development trends and future research directions in the field
- The abstract should be concise and clear, typically 200-300 words
- List 3-6 keywords (Keywords) at the end of the abstract, including the main concepts covered in the paper

## 1. Introduction
The introduction should be straightforward and concise, typically 200-300 words, aiming to quickly engage readers and generate interest in continuing to read. The introduction generally includes:
- Elucidate the background and importance of the research field
- Briefly introduce the historical development and current research status of the field
- Clearly state the research questions or objectives of the review, summarize the core problems and research motivations in the field
- Briefly describe the development trends and future research directions in the field
- Define the scope of the review (time, region, discipline, methods, etc.), explain the organizational structure of the review

For systematic or semi-systematic reviews, provide detailed information about literature search and screening methods, including:
- Describe the databases used (e.g., OpenAlex, arXiv)
- Explain search keywords, search queries, time range, etc.
- Clearly state the criteria for screening literature (inclusion and exclusion criteria)
- Briefly describe the process of literature screening
- If applicable, explain methods for assessing literature quality

At the end of the introduction section, provide a comprehensive discussion of the main literature and studies to be covered in this review, including:
- Clearly list the main literature and studies to be discussed in this review
- Briefly explain how the literature is organized (e.g., by theme, methodology, time period, etc.)
- Describe the time span, types of studies, main research questions covered, etc.
- If literature is organized by clusters, explain what research direction each cluster represents and the main studies included in each cluster

## 2. Basic Concepts and Theoretical Background
This section should provide detailed introductions to the main concepts and theoretical foundations covered in the provided literature, organized into subsections. Each subsection should focus on one core concept or theoretical framework.

**IMPORTANT: Comprehensive Synthesis Principle (Special Emphasis for This Section)**
- **Before writing each subsection, you MUST first comprehensively read all papers related to that concept or theory**
- **Do NOT introduce a concept after seeing only one paper; you must comprehensively analyze multiple related papers**
- **Identify common viewpoints, different viewpoints, and synthesized perspectives from these papers, then conduct comprehensive synthesis**
- **Each concept introduction MUST be based on comprehensive analysis of multiple papers, ensuring comprehensiveness and representativeness of viewpoints**
- **Avoid partiality: Do not introduce a concept or theory based solely on a single paper; cite multiple papers to support synthesized perspectives**

**Organization:**
- Carefully read the provided literature to identify all main concepts, theoretical frameworks, methodologies, technical principles, etc. covered in the articles
- Based on the identified concepts and theories, divide into multiple subsections (e.g., 2.1, 2.2, 2.3, etc.)
- Each subsection focuses on one core concept or theoretical background
- Use clear subheadings, directly using the names of concepts or theories, for example:
  * "2.1 Core Concept X"
  * "2.2 Theoretical Framework Y"
  * "2.3 Methodology Z"
  * "2.4 Technology W"
  * Or create subheadings based on the actual names of main concepts and theories covered in the literature (only use the names of concepts or theories, do not add descriptive words like "definition", "development", "principles", etc.)

**Each subsection should include (MUST be based on comprehensive analysis of multiple papers):**
- **Definition of concept/theory**: Clearly define the concept or theory. You MUST synthesize definitions from multiple papers, identify common points, and explain different perspectives if they exist
- **Historical development**: Introduce the proposal and development process of the concept or theory. You MUST comprehensively synthesize multiple relevant papers, thoroughly trace the development trajectory, and cite multiple papers
- **Core content**: Explain the core content, key elements, and basic principles of the concept or theory. You MUST base this on comprehensive analysis of multiple papers, identifying common core viewpoints
- **Theoretical relationships**: Explain relationships and evolution with other concepts or theories. You MUST comprehensively analyze perspectives from relevant papers
- **Application areas**: Introduce the application of the concept or theory in the research field. You MUST cite multiple papers and comprehensively introduce application situations
- **Main viewpoints**: Summarize the main viewpoints in the literature regarding this concept or theory. You MUST comprehensively analyze all relevant papers, identifying common viewpoints, different viewpoints, and synthesized perspectives
- **MUST cite multiple relevant papers to support the introduction of each concept or theory; do NOT cite only one paper**

**Main Body Organization Instructions:**
The main body is the core of the literature review and should be organized by themes or methodological approaches. Based on literature clusters or research themes, create independent chapters for each theme (starting from Chapter 3). Each chapter should use clear, informative headings, for example:
- "Early Conceptual Models of X" (historical evolution type)
- "Experimental Studies of Y" (research method type)
- "Qualitative Perspectives on Z" (methodological type)
- "Methodological Challenges in Measuring X" (methodological type)
- "Cross-Cultural Comparisons and Contextual Effects" (comparative research type)
- Or create headings based on research direction names provided in cluster titles

**Each theme chapter should include:**
- **Main points**: Present the main points and findings this section aims to convey
- **Contribution to overall argument**: Explain how this section contributes to the overall argument or objectives
- Summarize, compare, and critically analyze related research
- **STRICTLY PROHIBIT simple listing of papers**: Do NOT introduce papers in the theme one by one, do NOT write in the form of "Paper A did X, Paper B did Y"
- **MUST summarize and analyze everyone's approaches**: Comprehensively analyze research methods, research approaches, and research results from multiple papers in this theme, summarize:
  * Common research methods and approaches in this theme
  * Different research methods and approaches and their advantages/disadvantages
  * Development trends in research methods and approaches
  * Main research directions and hot topics in current research on this theme
- Cite representative literature and point out their contributions and limitations
- Discuss in detail the relevant studies in that cluster or theme, elaborating on research findings, methods, and contributions, but based on comprehensive analysis, not simple listing

**Note:** If literature is organized by clusters, each cluster represents a research theme, and you should create an independent chapter for each cluster (starting from Chapter 3, numbered as 3, 4, 5...). Please use the research direction names provided in the cluster titles as chapter headings, do NOT use "Cluster 0", "Cluster 1" or other numeric labels. Each chapter should:
1. Discuss all relevant studies in that cluster
2. Present the main points and findings of that research direction
3. Explain how that research direction contributes to the overall objectives of the review
4. If the theme involves historical evolution, use vertical comparison and analysis to analyze the process and context of the proposal and development of problems or theories
5. If the theme involves current research status, use horizontal analysis to compare current or recent literature, revealing unsolved problems and hot issues
6. If the theme involves future predictions, predict possible future development directions based on historical and current literature

**Optional content:**
- If helpful for readers, you may summarize the historical development of the field in a theme chapter
- Consider presenting a conceptual framework showing relationships between theories or variables
- Consider creating tables comparing key characteristics of major studies

## [N]. Discussion
The discussion section should provide comprehensive analysis, interpretation, and evaluation of the content in the main body:
- **Synthesize main findings**: Integrate the main findings from all subsections in the main body
- **Interpretation and significance**: Interpret the significance of research findings, analyze connections and differences between different studies
- **Theoretical contributions**: Discuss contributions of research to theoretical development
- **Methodological evaluation**: Evaluate methodological strengths and limitations of existing research
- **Research gaps**: Identify deficiencies and gaps in current research
- **Controversies and disagreements**: Analyze controversies and disagreements between different scholars or studies

## [N+1]. Conclusion
- **Synthesize core findings**: Summarize the main findings and contributions of the review
- **Emphasize overall progress**: Outline the overall development status of the field
- **Reiterate value**: Explain the significance and value of the review

## References
- List all cited literature in **APA 7th edition format** that are actually cited in the text
- **Sorting method**: **Arrange by author surname in alphabetical order**
- Ensure coverage of classic literature and the latest achievements from the past 3-5 years (if the provided literature includes such works)
- **Include only literature explicitly cited in the text**, do not include uncited references
- Each reference must include: Authors, Year, Title, Source (Journal name/Conference name/arXiv ID, etc.)

{citation_rules}

Writing Requirements:

**Important Principles for Literature Review Writing (MUST strictly follow):**

**Core Principle: Comprehensive Analysis First, Then Synthesized Generation**
- **Step 1: Comprehensive Analysis of All Papers (MUST complete before writing)**
  - MUST carefully read all provided paper abstracts
  - Identify all major themes, concepts, theories, and methods in all papers
  - Find common points, differences, and synthesized perspectives among all papers
  - Form a comprehensive, synthesized understanding of the field
  - Determine the organizational framework and logical structure of the review
  
- **Step 2: Generate Review Based on Comprehensive Analysis Results (Can only start after completing Step 1)**
  - All viewpoints must be based on the comprehensive analysis results from Step 1
  - Every viewpoint must be supported by multiple papers, not just a single paper
  - MUST reflect common points and differences among papers
  - **STRICTLY PROHIBIT simple listing of papers**: Do NOT introduce papers one by one, do NOT write in the form of "Paper A did X, Paper B did Y"
  - **MUST summarize and analyze everyone's approaches**: Comprehensively analyze research methods, research approaches, and research results from multiple papers, summarize:
    * Common research methods and approaches
    * Different research methods and approaches
    * Advantages and disadvantages of various methods
    * Development trends in research methods and approaches
  - MUST form synthesized discussions, not simply list viewpoints from individual papers
  - MUST ensure comprehensiveness and representativeness of viewpoints

**Special Emphasis:**
- **This is particularly important in Section 2 (Basic Concepts and Theoretical Background)**. When introducing each concept or theory, you MUST:
  1. First comprehensively read all papers related to that concept or theory
  2. Identify common viewpoints, different viewpoints, and synthesized perspectives from these papers
  3. Conduct comprehensive analysis before expression, ensuring comprehensiveness and accuracy of viewpoints
  4. Avoid introducing a concept by citing only one paper; you should cite multiple papers to support synthesized perspectives
- **In all sections, you MUST follow the principle of "comprehensive analysis first, then synthesized generation"**

**Format and Structure Requirements:**
- **MUST write the entire review in English, including all sections, citations, and references**
- **MUST generate the title at the very first line of the output, using `# **[Article Title]**` format, the title MUST be bold**
- **All first-level headings (section headings) MUST be bold, using format `## **1. Introduction**`, `## **2. Basic Concepts and Theoretical Background**`, etc.**
- Strictly follow the above structure: Title (no number), Abstract (no number), Section 1 (Introduction and Methods / Search Strategy), Section 2 (Basic Concepts and Theoretical Background), Sections 3-N (theme chapters dynamically created based on cluster count, one independent chapter per cluster), Discussion chapter, Conclusion chapter, References (no number), do not omit any part
- **Theme chapter organization**: Based on literature clustering results, create an independent chapter for each cluster (starting from Chapter 3). If literature is not clustered, organize theme chapters based on research themes, methodological approaches, or temporal dimensions
- **Theme chapter headings**: Use clear, informative headings, with each heading clearly indicating which studies will be discussed, main points, and contribution to overall argument
- **Logical progression**: Ensure clear logical progression from background and definitions, through evidence and analysis, to interpretation and significance
- Maintain clear logic and natural transitions between sections
- Do NOT output excessive blank lines, keep the format compact

**Content Quality Requirements:**
- Ensure the review is thorough, accurate, and insightful, demonstrating deep understanding of the field
- Write comprehensively with rich content, provide detailed analysis for each topic, and offer thorough explanations and discussions
- **STRICTLY PROHIBIT simple listing of papers**:
  - Do NOT introduce papers one by one, do NOT write in the form of "Author A et al. (2020) studied X, Author B et al. (2021) studied Y"
  - Do NOT introduce what each paper did one by one, but instead comprehensively analyze approaches from multiple papers
  - MUST summarize and analyze everyone's approaches: Comprehensively analyze research methods, research approaches, and research results from multiple papers, summarize:
    * Common research methods and approaches
    * Different research methods and approaches
    * Advantages and disadvantages of various methods
    * Development trends in research methods and approaches
    * Main research directions and hot topics in current research
- **MUST strictly follow citation rules: cite papers at appropriate places in EVERY section, do NOT omit citations**
- **Every major point, research finding, or technical method MUST be supported by relevant citations**
- **Make full use of the abstract information from each paper, accurately reflecting the research content, methods, and findings described in the abstracts**
- **For papers organized by clusters, reflect the differences and connections between research themes of different clusters in the review**
- Use author-year format citations at appropriate places, in formats: (Author, Year) or Author (Year)"""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_REASONING_MODEL", "deepseek-reasoner"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    temperature=0.3,
                    stream=True
                )

                full_text = ""

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                elapsed_time = first_token_time - start_time
                                print(f"[literature_review] 从开始到第一次生成文字的时间: {elapsed_time:.2f} 秒")
                            
                            full_text += delta_content
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"

                if papers_references:
                    cited_refs = []

                    if full_text:

                        for ref in papers_references:
                            author_surname = ref.get('author_surname', 'Unknown')
                            year = ref.get('year', '')
                            multiple_authors = ref.get('multiple_authors', False)
                            title = ref.get('title', '').lower()

                            patterns = []
                            if year:
                                year_str = str(year) if not isinstance(year, str) else year
                                if multiple_authors:
                                    # 英文引用格式
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?,? {year_str})")
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?, {year_str})")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? \\({year_str}\\)")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? {year_str}")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.?, {year_str}")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? {year_str}")
                                    patterns.append(f"（{re.escape(author_surname)} et al\\.?,? {year_str}）")
                                    patterns.append(f"（{re.escape(author_surname)}等,? {year_str}）")
                                    patterns.append(f"{re.escape(author_surname.lower())} et al\\.?.*?{year_str}")
                                    patterns.append(f"{re.escape(author_surname.upper())} et al\\.?.*?{year_str}")
                                else:
                                    patterns.append(f"({re.escape(author_surname)},? {year_str})")
                                    patterns.append(f"({re.escape(author_surname)} {year_str})")
                                    patterns.append(f"{re.escape(author_surname)} \\({year_str}\\)")
                                    patterns.append(f"{re.escape(author_surname)} {year_str}")
                                    patterns.append(f"{re.escape(author_surname)}, {year_str}")
                                    patterns.append(f"（{re.escape(author_surname)},? {year_str}）")
                                    patterns.append(f"（{re.escape(author_surname)} {year_str}）")
                                    patterns.append(f"{re.escape(author_surname.lower())}.*?{year_str}")
                                    patterns.append(f"{re.escape(author_surname.upper())}.*?{year_str}")
                            else:
                                if multiple_authors:
                                    patterns.append(f"{re.escape(author_surname)} et al\\.")
                                    patterns.append(f"{re.escape(author_surname)}等")
                                    patterns.append(f"{re.escape(author_surname.lower())} et al\\.")
                                    patterns.append(f"{re.escape(author_surname.upper())} et al\\.")
                                else:
                                    patterns.append(re.escape(author_surname))
                                    patterns.append(re.escape(author_surname.lower()))
                                    patterns.append(re.escape(author_surname.upper()))

                            cited = False
                            for pattern in patterns:
                                if re.search(pattern, full_text, re.IGNORECASE):
                                    cited = True
                                    break

                            if not cited and title:
                                if any('\u4e00' <= char <= '\u9fff' for char in title):
                                    title_keywords = title[:10] if len(title) >= 10 else title
                                    if len(title_keywords) > 3 and title_keywords in full_text.lower():
                                        cited = True
                                else:
                                    title_words = title.split()[:5]
                                    if title_words:
                                        important_words = [w for w in title_words if len(w) > 3]
                                        if important_words:
                                            found_count = sum(1 for word in important_words if
                                                              re.search(r'\b' + re.escape(word) + r'\b', full_text,
                                                                        re.IGNORECASE))
                                            if found_count >= min(3, len(important_words)):
                                                cited = True

                            if cited:
                                cited_refs.append(ref)
                    else:
                        cited_refs = papers_references

                    # 检查是否有英文文献未被检测到
                    english_refs = [ref for ref in papers_references if ref.get('author_surname', '') and not any('\u4e00' <= char <= '\u9fff' for char in ref.get('author_surname', ''))]
                    english_cited = [ref for ref in cited_refs if ref.get('author_surname', '') and not any('\u4e00' <= char <= '\u9fff' for char in ref.get('author_surname', ''))]
                    
                    if len(english_refs) > 0:
                        english_cited_ratio = len(english_cited) / len(english_refs) if len(english_refs) > 0 else 0
                        
                        if english_cited_ratio < 0.7 and len(english_refs) > 0:
                            cited_refs = papers_references
                    
                    if len(cited_refs) < len(papers_references) * 0.5 and len(papers_references) > 0:
                        cited_refs = papers_references
                    elif len(cited_refs) == 0 and len(papers_references) > 0:
                        cited_refs = papers_references

                yield "data: [DONE]\n\n"

            except asyncio.TimeoutError:
                error_data = {
                    "object": "error",
                    "message": "Request timeout (15 minutes limit)"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[literature_review] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    Paper Q&A endpoint - uses reasoning model with PDF content
    
    Request body:
    {
        "query": "Please carefully analyze and explain the reinforcement learning training methods used in this article.",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        language = detect_language(query)

        async def generate():
            try:
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    error_data = {
                        "object": "error",
                        "message": error_msg
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if len(pdf_text) > 8000:
                    rag = SimpleRAG()
                    relevant_chunks = await rag.retrieve_relevant_chunks(query, pdf_text, top_k=5)
                    context = "\n\n".join(relevant_chunks)
                    paper_content = context
                else:
                    paper_content = pdf_text

                if language == 'zh':
                    prompt = f"""请基于以下论文内容回答问题。

论文内容:

{paper_content}

问题: {query}

请仔细分析论文内容，准确回答问题。如果论文中没有相关信息，请说明。使用Markdown格式输出答案。"""
                else:
                    prompt = f"""Answer the question based on the paper content.

Paper:

{paper_content}

Question: {query}"""

                stream = await reasoning_client.chat.completions.create(
                    model=os.getenv("SCI_LLM_REASONING_MODEL", "deepseek-reasoner"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.2,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta

                        reasoning_content = getattr(delta, 'reasoning_content', None)

                        delta_content = delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"[paper_qa] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/ideation")
async def ideation(request: Request):
    """
    Ideation endpoint - uses embedding model for similarity and LLM for generation
    
    Request body:
    {
        "query": "Generate research ideas about climate change"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        reference_ideas = [
            "Using deep learning to predict protein folding structures",
            "Applying transformer models to drug discovery and molecular design",
            "Leveraging reinforcement learning for automated experiment design",
            "Developing AI-powered literature review and knowledge synthesis tools",
            "Creating neural networks for climate modeling and weather prediction",
            "Using machine learning to analyze large-scale genomic datasets"
        ]

        language = detect_language(query)

        async def generate():
            try:
                if language == 'zh':
                    prompt = f"""为以下研究领域生成创新的研究想法：

{query}"""
                else:
                    prompt = f"""Generate innovative research ideas for:

{query}"""

                query_embedding = await get_embedding(query)

                if query_embedding:
                    similarities = []
                    for idx, idea in enumerate(reference_ideas):
                        idea_embedding = await get_embedding(idea)
                        if idea_embedding:
                            similarity = cosine_similarity(query_embedding, idea_embedding)
                            similarities.append((idx, idea, similarity))

                    similarities.sort(key=lambda x: x[2], reverse=True)

                    if language == 'zh':
                        prompt += f"\n\n参考想法（按相似度排序）：\n"
                    else:
                        prompt += f"\n\nReference ideas (ranked by similarity):\n"

                    for idx, idea, sim in similarities:
                        prompt += f"\n{idx + 1}. (similarity: {sim:.3f}) {idea}"

                    if language == 'zh':
                        prompt += "\n\n基于以上参考想法生成新颖的研究想法。"
                    else:
                        prompt += "\n\nGenerate novel research ideas based on the above."

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"[ideation] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses LLM model with PDF content
    
    Request body:
    {
        "query": "Please review this paper",  # optional, default review prompt will be used
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "Please provide a comprehensive review of this paper")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        language = detect_language(query)

        async def generate():
            try:
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    error_data = {
                        "object": "error",
                        "message": error_msg
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if language == 'zh':
                    prompt = f"""评审以下论文：

论文内容:

{pdf_text}

指令: {query}"""
                else:
                    prompt = f"""Review the following paper:

Paper:

{pdf_text}

Instruction: {query}"""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"[paper_review] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
