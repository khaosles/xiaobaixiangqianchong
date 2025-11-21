import httpx
from typing import List, Optional
from base import BaseSearcher, PaperResult


class OpenAlexSearcher(BaseSearcher):
    """
    OpenAlex 文献搜索器
    """
    
    def __init__(self):
        super().__init__("OpenAlex")
    
    @staticmethod
    def decode_abstract(inverted_index: Optional[dict]) -> Optional[str]:
        """
        将 OpenAlex 的 inverted_index 解码为原始摘要文本
        
        Args:
            inverted_index: OpenAlex 返回的倒排索引格式的摘要
            
        Returns:
            解码后的摘要文本
        """
        if not inverted_index:
            return None

        try:
            # 创建一个足够长的列表来存放单词
            max_position = max(pos for positions in inverted_index.values() for pos in positions)
            word_list = [''] * (max_position + 1)

            for word, positions in inverted_index.items():
                for pos in positions:
                    word_list[pos] = word
            return ' '.join(word_list)
        except Exception:
            return None
    
    async def search(self, query: str, max_results: int = 10, sorted: Optional[str] = None, **kwargs) -> List[PaperResult]:
        """
        搜索 OpenAlex 文献并获取可读摘要（异步方法）
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sorted: 排序方式，可选值：
                - "relevance" 或 None: 按相关性排序（默认）
                - "date" 或 "publication_date:desc": 按发表日期降序
                - "cited_by_count:desc": 按引用次数降序
                - "publication_date:desc,cited_by_count:desc": 先按日期再按引用次数
            **kwargs: 其他参数（当前未使用）
            
        Returns:
            论文结果列表
            :param query:
            :param max_results:
            :param sorted:
            :param kwargs:
            :return:
        """
        url = "https://api.openalex.org/works"
        
        # 处理排序参数
        sort_param = None
        if sorted:
            if sorted == "relevance":
                sort_param = None  # OpenAlex 默认按相关性排序
            elif sorted == "date":
                sort_param = "publication_date:desc"
            elif sorted.startswith("publication_date") or sorted.startswith("cited_by_count"):
                sort_param = sorted
            else:
                # 默认使用提供的排序字符串
                sort_param = sorted
        else:
            # 默认排序：先按日期再按引用次数
            sort_param = 'publication_date:desc,cited_by_count:desc'
        
        params = {
            'search': query,
            'per-page': max_results,
            'page': 1,
            'select': 'id,title,doi,publication_year,abstract_inverted_index,authorships',
            "mailto": "khaosles@163.com"
        }
        
        if sort_param:
            params['sort'] = sort_param

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    print(f"OpenAlex 请求失败: {response.status_code}")
                    return []
                data = response.json()
        except Exception as e:
            print(f"OpenAlex 请求异常: {e}")
            return []
        results = []

        for item in data.get('results', []):
            # 解码摘要
            abstract_inv = item.get('abstract_inverted_index')
            abstract = self.decode_abstract(abstract_inv)
            
            # 提取作者
            authors = []
            authorships = item.get('authorships', [])
            for authorship in authorships:
                author = authorship.get('author', {})
                display_name = author.get('display_name')
                if display_name:
                    authors.append(display_name)
            
            # 提取 DOI（去掉前缀）
            doi = item.get('doi')
            if doi and doi.startswith('https://doi.org/'):
                doi = doi.replace('https://doi.org/', '')
            
            # 构建 URL
            url_str = None
            if doi:
                url_str = f"https://doi.org/{doi}"
            elif item.get('id'):
                url_str = item.get('id')
            
            # 创建结果对象
            paper = PaperResult(
                title=item.get('title', ''),
                authors=self._normalize_authors(authors),
                abstract=abstract,
                year=item.get('publication_year'),
                doi=doi,
                url=url_str,
                pdf_url=None,
                source=self.source_name,
                published_date=str(item.get('publication_year', '')) if item.get('publication_year') else None
            )
            
            results.append(paper)

        return results


# 向后兼容的函数接口（异步）
async def search_works_with_abstract(query: str, max_results: int = 10) -> List[dict]:
    """
    向后兼容的函数接口（异步）
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数
        
    Returns:
        旧格式的字典列表
    """
    searcher = OpenAlexSearcher()
    papers = await searcher.search(query, max_results)
    
    # 转换为旧格式
    results = []
    for paper in papers:
        results.append({
            'title': paper.title,
            'doi': paper.doi,
            'year': paper.year,
            'abstract': paper.abstract
        })
    
    return results


# 使用示例
if __name__ == "__main__":
    import asyncio
    
    async def main():
        searcher = OpenAlexSearcher()
        papers = await searcher.search("叶面积指数", max_results=20, sorted='cited_by_count:desc')

        for i, p in enumerate(papers, 1):
            print(f"论文 {i}: {p.title}")
            print(f"作者: {', '.join(p.authors) if p.authors else '未知'}")
            print(f"年份: {p.year}")
            print(f"DOI: {p.doi}")
            print(f"URL: {p.url}")
            print("摘要:")
            if p.abstract:
                print(p.abstract[:500] + "..." if len(p.abstract) > 500 else p.abstract)
            else:
                print("（无摘要）")
            print("-" * 80)
    
    asyncio.run(main())
