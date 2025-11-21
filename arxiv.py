import httpx
import urllib.parse
from xml.etree import ElementTree as ET
from datetime import datetime
from typing import List, Optional
from base import BaseSearcher, PaperResult


class ArxivSearcher(BaseSearcher):
    """
    arXiv 文献搜索器
    """
    
    def __init__(self):
        super().__init__("arXiv")
    
    async def search(self, query: str, max_results: int = 10, sorted: Optional[str] = None, **kwargs) -> List[PaperResult]:
        """
        使用 arXiv API 搜索论文（异步方法）
        
        Args:
            query: 搜索关键词，如 "large language models"
            max_results: 最多返回多少篇
            sorted: 排序方式，可选值：
                - "relevance" 或 None: 按相关性排序（默认）
                - "date" 或 "submittedDate": 按提交日期排序
                - "lastUpdatedDate": 按最后更新时间排序
            **kwargs: 其他参数，支持 sort_by（向后兼容，会被 sorted 覆盖）
            
        Returns:
            论文结果列表
        """
        # 处理排序参数（支持 sorted 和向后兼容的 sort_by）
        sort_by = kwargs.get('sort_by', None)
        if sorted is None:
            sorted = sort_by if sort_by else "relevance"
        
        # 将简化的排序选项转换为 arXiv API 格式
        # arXiv API 支持的 sortBy 值: relevance, lastUpdatedDate, submittedDate
        if sorted == "date":
            sort_by_param = "submittedDate"
        elif sorted in ["relevance", None]:
            sort_by_param = "relevance"  # 使用标准的 relevance，而不是 relevanceLastAuthorDate
        elif sorted == "lastUpdatedDate":
            sort_by_param = "lastUpdatedDate"
        else:
            # 如果提供了其他值，尝试直接使用（可能是有效的 arXiv API 参数）
            sort_by_param = sorted
        
        # arXiv API 要求对查询词 URL 编码
        encoded_query = urllib.parse.quote(query)
        url = (
            f"https://export.arxiv.org/api/query?"
            f"search_query=all:{encoded_query}&"
            f"start=0&"
            f"max_results={max_results}&"
            f"sortBy={sort_by_param}&"
            f"sortOrder=descending"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.content
        except httpx.RequestError as e:
            print(f"arXiv 请求失败: {e}")
            return []
        except httpx.HTTPStatusError as e:
            print(f"arXiv HTTP 错误: {e}")
            return []

        # 解析 XML
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            print(f"arXiv XML 解析失败: {e}")
            return []

        # 命名空间
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        papers = []
        for entry in root.findall('atom:entry', ns):
            # 标题（去除换行）
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "无标题"

            # 摘要
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip() if summary is not None else None

            # 作者
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)

            # 发表日期（published）
            published = entry.find('atom:published', ns)
            pub_date = None
            year = None
            if published is not None:
                try:
                    pub_date = datetime.strptime(published.text, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                    year = int(published.text[:4])
                except:
                    pub_date = published.text
                    year = self._extract_year(pub_date)

            # arXiv ID 和 PDF 链接
            arxiv_id = None
            pdf_url = None
            abs_url = None
            for link in entry.findall('atom:link', ns):
                href = link.get('href')
                if href and 'arxiv.org/abs/' in href:
                    arxiv_id = href.split('/')[-1]
                    abs_url = href
                if link.get('title') == 'pdf':
                    pdf_url = href

            # 分类（primary category）
            primary_cat = entry.find('arxiv:primary_category', ns)
            category = primary_cat.get('term') if primary_cat is not None else None

            # 创建结果对象
            paper = PaperResult(
                title=title,
                authors=self._normalize_authors(authors if authors else ['未知作者']),
                abstract=abstract,
                year=year,
                doi=None,  # arXiv 通常没有 DOI
                url=abs_url,
                pdf_url=pdf_url,
                source=self.source_name,
                arxiv_id=arxiv_id,
                category=category,
                published_date=pub_date
            )
            
            papers.append(paper)

        return papers


# 向后兼容的函数接口（异步）
async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> List[dict]:
    """
    向后兼容的函数接口（异步）
    
    Args:
        query: 搜索关键词
        max_results: 最多返回多少篇
        sort_by: 排序方式
        
    Returns:
        旧格式的字典列表
    """
    searcher = ArxivSearcher()
    papers = await searcher.search(query, max_results, sorted=sort_by)
    
    # 转换为旧格式
    results = []
    for paper in papers:
        results.append({
            'title': paper.title,
            'authors': paper.authors,
            'abstract': paper.abstract,
            'published': paper.published_date,
            'arxiv_id': paper.arxiv_id,
            'category': paper.category,
            'pdf_url': paper.pdf_url,
            'abs_url': paper.url
        })
    
    return results


if __name__ == "__main__":
    import asyncio
    
    # 🔧 测试用例（可直接运行，无需命令行参数）
    async def main():
        query = "resnet"  # ← 在这里修改你的关键词
        limit = 1  # ← 修改返回结果数量

        print(f"正在 arXiv 搜索: 「{query}」 (最多 {limit} 篇)...\n")

        searcher = ArxivSearcher()
        papers = await searcher.search(query, max_results=limit, sorted="date")

        if not papers:
            print("未找到相关论文。")
        else:
            for i, p in enumerate(papers, 1):
                print(f"[{i}] {p.title}")
                print(f"   作者: {', '.join(p.authors)}")
                print(f"   分类: {p.category} | 日期: {p.published_date}")
                print(f"   页面: {p.url}")
                print(f"   PDF : {p.pdf_url}")
                print("   摘要:")
                if p.abstract:
                    print(f"   {p.abstract[:600]}{'...' if len(p.abstract) > 600 else ''}")
                else:
                    print("   （无摘要）")
                print("-" * 80)
    
    asyncio.run(main())
