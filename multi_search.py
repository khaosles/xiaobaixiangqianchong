"""
多源搜索工具
整合多个数据源的搜索结果，并按 DOI 去重
"""
import asyncio
from typing import List
from base import PaperResult
from openalex import OpenAlexSearcher
from arxiv import ArxivSearcher


async def multi_source_search(
    query: str,
    max_results_per_source: int = 1,
    **kwargs
) -> List[PaperResult]:
    """
    多源搜索方法，整合 OpenAlex 和 arXiv 的搜索结果
    
    搜索策略：
    1. OpenAlex 按引用量排序搜索
    2. OpenAlex 按相关性排序搜索
    3. OpenAlex 按出版时间排序搜索
    4. arXiv 搜索
    
    所有结果按 DOI 去重，保留第一个出现的论文
    
    Args:
        query: 搜索关键词
        max_results_per_source: 每个数据源的最大返回结果数
        **kwargs: 其他参数
        
    Returns:
        去重后的论文结果列表
    """
    # 创建搜索器实例
    openalex_searcher = OpenAlexSearcher()
    arxiv_searcher = ArxivSearcher()
    
    # 并行执行所有搜索任务
    tasks = [
        # OpenAlex 按引用量排序
        openalex_searcher.search(
            query, 
            max_results=max_results_per_source, 
            sorted="cited_by_count:desc,relevance_score:desc"
        ),
        # OpenAlex 按相关性排序
        openalex_searcher.search(
            query, 
            max_results=max_results_per_source, 
            sorted="relevance"
        ),
        # OpenAlex 按出版时间排序
        openalex_searcher.search(
            query, 
            max_results=max_results_per_source, 
            sorted="publication_date:desc,relevance_score:desc"
        ),
        # arXiv 搜索
        arxiv_searcher.search(
            query, 
            max_results=max_results_per_source
        ),
    ]
    
    # 等待所有搜索完成
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 合并所有结果
    all_papers = []
    for i, result in enumerate(results_list):
        if isinstance(result, Exception):
            print(f"搜索任务 {i+1} 失败: {result}")
            continue
        if isinstance(result, list):
            all_papers.extend(result)
    
    # 按 DOI 去重
    seen_dois = set()
    unique_papers = []
    
    for paper in all_papers:
        # 标准化 DOI（去掉前缀，转为小写）
        doi_key = None
        if paper.doi:
            doi_key = paper.doi.lower().strip()
            # 去掉可能的 URL 前缀
            if doi_key.startswith('https://doi.org/'):
                doi_key = doi_key.replace('https://doi.org/', '')
            elif doi_key.startswith('http://doi.org/'):
                doi_key = doi_key.replace('http://doi.org/', '')
            elif doi_key.startswith('doi:'):
                doi_key = doi_key.replace('doi:', '')
        
        # 如果论文有 DOI，检查是否已存在
        if doi_key:
            if doi_key in seen_dois:
                continue  # 跳过重复的 DOI
            seen_dois.add(doi_key)
        
        # 对于没有 DOI 的论文（如 arXiv），使用标题和第一作者作为去重依据
        if not doi_key:
            # 创建唯一标识：标题（小写）+ 第一作者（如果有）
            title_lower = paper.title.lower().strip() if paper.title else ""
            first_author = ""
            if paper.authors and len(paper.authors) > 0:
                first_author = paper.authors[0].lower().strip()
            
            unique_key = f"{title_lower}|{first_author}"
            if unique_key in seen_dois:
                continue
            seen_dois.add(unique_key)
        
        unique_papers.append(paper)
    
    return unique_papers


# 使用示例
if __name__ == "__main__":
    async def main():
        query = "resnet"
        results = await multi_source_search(query, max_results_per_source=1)
        
        print(f"搜索关键词: {query}")
        print(f"找到 {len(results)} 篇去重后的论文\n")
        
        for i, paper in enumerate(results, 1):
            print(f"{i}. {paper.title}")
            print(f"   来源: {paper.source}")
            if paper.authors:
                print(f"   作者: {', '.join(paper.authors[:3])}")
            if paper.year:
                print(f"   年份: {paper.year}")
            if paper.doi:
                print(f"   DOI: {paper.doi}")
            if paper.url:
                print(f"   URL: {paper.url}")
            print("-" * 80)
    
    asyncio.run(main())

