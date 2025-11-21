import os
import json
import re
import asyncio
from typing import List, Dict, Optional
import httpx
import PyPDF2
from io import BytesIO
import xml.etree.ElementTree as ET


class LiteratureSearcher:
    """文献搜索器"""

    def __init__(self):
        # 优化超时时间和连接池，提高响应速度
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),  # 连接超时5秒，总超时10秒
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15)  # 增加连接数
        )

    async def search_arxiv(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Dict]:
        """
        从arXiv搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sort_by: 排序方式，"relevance" 按相关性排序，"date" 按时间排序
            
        Returns:
            文献列表，每个包含title, authors, abstract, pdf_url等信息
        """
        try:
            url = "https://export.arxiv.org/api/query"
            if sort_by == "date":
                sort_param = "submittedDate"
            else:
                sort_param = "relevanceLastAuthorDate"
            
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_param,
                "sortOrder": "descending"
            }

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            papers = []
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'source': 'arXiv',
                    'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else '',
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    'abstract': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                    'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                    'pdf_url': None,
                    'url': None,
                    'doi': None,  # arXiv通常没有DOI
                    'referenced_works': []  # arXiv API不提供引用信息
                }

                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        paper['pdf_url'] = link.get('href')
                    elif link.get('rel') == 'alternate':
                        paper['url'] = link.get('href')

                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1] if entry.find('atom:id', ns) is not None else ''
                paper['arxiv_id'] = arxiv_id

                papers.append(paper)

            return papers

        except Exception as e:
            print(f"arXiv搜索错误: {e}")
            return []

    async def search_openalex(self, query: str, max_results: int = 10, sort_by: str = "relevance", include_references: bool = False) -> List[Dict]:
        """
        从OpenAlex搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sort_by: 排序方式，"relevance" 按相关性排序，"date" 按时间排序，"citations" 按引用次数排序
            include_references: 是否获取引用文献列表，默认为False
            
        Returns:
            文献列表
        """
        try:
            url = "https://api.openalex.org/works"
            if sort_by == "citations":
                sort_param = "cited_by_count:desc"
            elif sort_by == "date":
                sort_param = "publication_date:desc"
            else:
                sort_param = None
            
            params = {
                "search": query,
                "per_page": min(max_results, 200),
                'select': 'id,title,doi,publication_year,abstract_inverted_index'
            }
            if sort_param:
                params["sort"] = sort_param
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            # 先收集所有work信息，然后批量获取摘要
            works_data = []
            for work in data.get("results", [])[:max_results]:
                # 处理OpenAlex的摘要格式（可能是字符串或inverted_index格式）
                abstract = ""
                if work.get("abstract"):
                    if isinstance(work.get("abstract"), str):
                        abstract = work.get("abstract")
                    elif isinstance(work.get("abstract"), dict) and "inverted_index" in work.get("abstract", {}):
                        inverted_index = work.get("abstract", {}).get("inverted_index", {})
                        if inverted_index:
                            words = []
                            for word, positions in inverted_index.items():
                                for pos in positions:
                                    words.append((pos, word))
                            words.sort(key=lambda x: x[0])
                            abstract = " ".join([w[1] for w in words])
                
                paper = {
                    "source": "OpenAlex",
                    "title": work.get("title", ""),
                    "authors": [author.get("author", {}).get("display_name", "") 
                               for author in work.get("authorships", [])],
                    "abstract": abstract,
                    "published": work.get("publication_date", ""),
                    "doi": work.get("doi", ""),
                    "url": work.get("primary_location", {}).get("landing_page_url") or work.get("id"),
                    "pdf_url": None,
                    "openalex_id": work.get("id", "").split("/")[-1] if work.get("id") else "",
                    "referenced_works": []  # 引用的文献列表（OpenAlex ID）
                }
                
                # 获取引用的文献信息（如果启用）
                if include_references:
                    referenced_works = work.get("referenced_works", [])
                    if referenced_works:
                        paper["referenced_works"] = referenced_works[:50]  # 限制前50个引用
                
                # 如果没有摘要，记录需要补充的文献
                if not abstract:
                    works_data.append((paper, work))
                
                for location in work.get("locations", []):
                    if location.get("pdf_url"):
                        paper["pdf_url"] = location["pdf_url"]
                        break
                
                if not paper["pdf_url"]:
                    oa_info = work.get("open_access", {})
                    if oa_info.get("oa_url"):
                        paper["pdf_url"] = oa_info["oa_url"]
                
                papers.append(paper)
            
            # 批量获取缺失的摘要（使用多源备用方案）
            if works_data:
                tasks = []
                for paper, work in works_data:
                    # 优先从OpenAlex获取摘要
                    if work.get("doi"):
                        tasks.append(("openalex", paper, work.get("doi")))
                    elif work.get("id"):
                        work_id = work.get("id").split("/")[-1] if "/" in work.get("id", "") else work.get("id")
                        tasks.append(("openalex_id", paper, work_id))
                
                if tasks:
                    # 优化并发数量，提高摘要获取速度
                    semaphore = asyncio.Semaphore(10)  # 增加并发数从5到10
                    async def fetch_abstract(task_item):
                        async with semaphore:
                            source_type, paper, identifier = task_item
                            # 先尝试从OpenAlex获取
                            if source_type == "openalex":
                                abstract = await self._get_abstract_from_openalex_by_doi(identifier)
                                if abstract:
                                    return paper, abstract
                                # 如果OpenAlex没有，直接使用多源备用方案（不重复请求OpenAlex）
                                abstract = await self._get_abstract_by_doi(identifier)
                                return paper, abstract
                            elif source_type == "openalex_id":
                                abstract = await self._get_abstract_from_openalex_by_work_id(identifier)
                                if abstract:
                                    return paper, abstract
                                # 如果OpenAlex没有，且paper有doi，尝试多源备用方案
                                if paper.get("doi"):
                                    abstract = await self._get_abstract_by_doi(paper.get("doi"))
                                    return paper, abstract
                            return paper, ""
                    
                    limited_tasks = [fetch_abstract(task_item) for task_item in tasks]
                    results = await asyncio.gather(*limited_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        if isinstance(result, tuple) and len(result) == 2:
                            paper, abstract = result
                            if abstract:
                                paper["abstract"] = abstract
            
            return papers
            
        except Exception as e:
            print(f"OpenAlex搜索错误: {e}")
            return []

    async def search_crossref(self, query: str, max_results: int = 10, sort_by: str = "relevance", include_references: bool = False) -> List[Dict]:
        """
        从Crossref搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sort_by: 排序方式，"relevance" 按相关性排序，"date" 按时间排序，"citations" 按引用次数排序
            include_references: 是否获取引用文献列表，默认为False
            
        Returns:
            文献列表
        """
        try:
            url = "https://api.crossref.org/works"
            if sort_by == "citations":
                sort_param = "is-referenced-by-count"
            elif sort_by == "date":
                sort_param = "date"
            else:
                sort_param = "relevance"
            
            params = {
                "query": query,
                "rows": min(max_results, 1000),  # Crossref允许最多1000
                "sort": sort_param,
                "order": "desc"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            # 先收集所有文献信息，然后批量获取摘要
            crossref_papers = []
            for item in data.get("message", {}).get("items", [])[:max_results]:
                # 尝试从Crossref获取摘要（Crossref很少提供摘要）
                abstract = ""
                abstract_data = item.get("abstract")
                if abstract_data:
                    if isinstance(abstract_data, str):
                        abstract = abstract_data
                    elif isinstance(abstract_data, dict):
                        abstract = abstract_data.get("text", "") or str(abstract_data)
                
                paper = {
                    "source": "Crossref",
                    "title": " ".join(item.get("title", [])),
                    "authors": [f"{author.get('given', '')} {author.get('family', '')}".strip()
                               for author in item.get("author", [])],
                    "abstract": abstract,
                    "published": item.get("published-print", {}).get("date-parts", [[None]])[0][0] if item.get("published-print") else 
                                item.get("published-online", {}).get("date-parts", [[None]])[0][0] if item.get("published-online") else None,
                    "doi": item.get("DOI", ""),
                    "url": f"https://doi.org/{item.get('DOI', '')}" if item.get("DOI") else None,
                    "pdf_url": None,
                    "referenced_works": []  # 引用的文献列表
                }
                
                # Crossref API中的引用文献信息在reference字段中（如果启用）
                if include_references:
                    references = item.get("reference", [])
                    if references:
                        referenced_dois = []
                        for ref in references[:50]:  # 限制前50个引用
                            ref_doi = ref.get("DOI", "")
                            if ref_doi:
                                referenced_dois.append(ref_doi)
                        paper["referenced_works"] = referenced_dois
                
                if paper["doi"]:
                    paper["url"] = f"https://doi.org/{paper['doi']}"
                
                papers.append(paper)
                # 如果没有摘要但有DOI，记录需要补充（在批量阶段统一处理）
                if not abstract and paper.get("doi"):
                    crossref_papers.append(paper)
            
            # 批量通过DOI获取摘要（使用多源备用方案）
            if crossref_papers:
                # 优化并发数量，提高摘要获取速度
                semaphore = asyncio.Semaphore(10)  # 增加并发数从5到10
                async def fetch_abstract_for_crossref(paper):
                    async with semaphore:
                        doi = paper.get("doi")
                        if not doi:
                            return paper, ""
                        # 先尝试从OpenAlex获取摘要
                        abstract = await self._get_abstract_from_openalex_by_doi(doi)
                        if abstract:
                            return paper, abstract
                        # 如果OpenAlex没有，使用多源备用方案（Crossref、Semantic Scholar）
                        abstract = await self._get_abstract_by_doi(doi)
                        return paper, abstract
                
                limited_tasks = [fetch_abstract_for_crossref(paper) for paper in crossref_papers]
                results = await asyncio.gather(*limited_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    if isinstance(result, tuple) and len(result) == 2:
                        paper, abstract = result
                        if abstract:
                            paper["abstract"] = abstract
            
            return papers
            
        except Exception as e:
            print(f"Crossref搜索错误: {e}")
            return []

    async def search_unpaywall(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        通过Unpaywall查找开放获取论文
        注意：Unpaywall主要用于通过DOI查找开放获取版本，这里结合其他数据源使用
        
        Args:
            query: 搜索关键词（Unpaywall主要用于DOI查找，这里先搜索其他源获取DOI）
            max_results: 最大返回结果数
            
        Returns:
            文献列表
        """
        try:
            crossref_results = await self.search_crossref(query, max_results)
            
            papers = []
            for paper in crossref_results:
                if paper.get("doi"):
                    unpaywall_url = f"https://api.unpaywall.org/v2/{paper['doi']}?email=openaccess@example.com"
                    try:
                        unpaywall_response = await self.client.get(unpaywall_url)
                        if unpaywall_response.status_code == 200:
                            unpaywall_data = unpaywall_response.json()
                            if unpaywall_data.get("is_oa") and unpaywall_data.get("best_oa_location"):
                                paper["pdf_url"] = unpaywall_data["best_oa_location"].get("url_for_pdf") or \
                                                  unpaywall_data["best_oa_location"].get("url_for_landing_page")
                                paper["source"] = "Unpaywall (via Crossref)"
                    except:
                        pass
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Unpaywall搜索错误: {e}")
            return []

    async def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """
        下载PDF文件
        
        Args:
            pdf_url: PDF文件URL
            
        Returns:
            PDF文件的字节内容，失败返回None
        """
        try:
            response = await self.client.get(pdf_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"下载PDF错误 {pdf_url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        从PDF内容中提取文本
        
        Args:
            pdf_content: PDF文件的字节内容
            
        Returns:
            提取的文本内容
        """
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_parts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"PDF文本提取错误: {e}")
            return ""

    async def _get_abstract_from_openalex_by_doi(self, doi: str) -> str:
        """
        通过DOI从OpenAlex获取摘要
        
        Args:
            doi: 文献的DOI
            
        Returns:
            摘要文本，如果获取失败返回空字符串
        """
        if not doi:
            return ""
        
        try:
            openalex_url = f"https://api.openalex.org/works/doi:{doi}"
            openalex_response = await self.client.get(openalex_url, timeout=5.0)  # 优化超时时间
            if openalex_response.status_code == 200:
                openalex_data = openalex_response.json()
                return self._extract_abstract_from_openalex_data(openalex_data)
        except:
            pass
        return ""

    async def _get_abstract_from_openalex_by_work_id(self, work_id: str) -> str:
        """
        通过work ID从OpenAlex获取摘要
        
        Args:
            work_id: OpenAlex的work ID
            
        Returns:
            摘要文本，如果获取失败返回空字符串
        """
        if not work_id:
            return ""
        
        try:
            detail_url = f"https://api.openalex.org/works/{work_id}"
            detail_response = await self.client.get(detail_url, timeout=5.0)  # 优化超时时间
            if detail_response.status_code == 200:
                detail_data = detail_response.json()
                return self._extract_abstract_from_openalex_data(detail_data)
        except:
            pass
        return ""

    def _extract_abstract_from_openalex_data(self, data: dict) -> str:
        """
        从OpenAlex数据中提取摘要
        
        Args:
            data: OpenAlex API返回的数据
            
        Returns:
            摘要文本，如果获取失败返回空字符串
        """
        if not data.get("abstract"):
            return ""
        
        abstract_obj = data.get("abstract")
        if isinstance(abstract_obj, str):
            return abstract_obj
        elif isinstance(abstract_obj, dict) and "inverted_index" in abstract_obj:
            inverted_index = abstract_obj.get("inverted_index", {})
            if inverted_index:
                words = []
                for word, positions in inverted_index.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort(key=lambda x: x[0])
                return " ".join([w[1] for w in words])
        return ""

    def _normalize_doi(self, doi: str) -> str:
        """标准化DOI格式"""
        if not doi:
            return ""
        
        # 移除可能的URL前缀
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
        elif doi.startswith("http://doi.org/"):
            doi = doi.replace("http://doi.org/", "")
        elif doi.startswith("doi:"):
            doi = doi.replace("doi:", "")
        
        return doi.strip()

    async def _get_abstract_by_doi(self, doi: str) -> str:
        """
        通过DOI从多个数据源获取摘要（备用方案）
        
        Args:
            doi: 标准化后的DOI
            
        Returns:
            摘要文本，如果获取失败返回空字符串
        """
        if not doi:
            return ""
        
        # 标准化DOI
        normalized_doi = self._normalize_doi(doi)
        if not normalized_doi:
            return ""
        
        # 尝试从Crossref获取摘要（优化：并行请求多个源，第一个成功即返回）
        tasks = []
        
        # Crossref任务
        async def fetch_crossref():
            try:
                crossref_url = f"https://api.crossref.org/works/{normalized_doi}"
                crossref_response = await self.client.get(crossref_url, timeout=4.0)
                if crossref_response.status_code == 200:
                    crossref_data = crossref_response.json()
                    message = crossref_data.get("message", {})
                    abstract_data = message.get("abstract")
                    if abstract_data:
                        if isinstance(abstract_data, str):
                            return abstract_data
                        elif isinstance(abstract_data, dict):
                            abstract_text = abstract_data.get("text", "")
                            if abstract_text:
                                return abstract_text
            except:
                pass
            return None
        
        # Semantic Scholar任务
        async def fetch_semantic_scholar():
            try:
                ss_url = f"https://api.semanticscholar.org/v1/paper/{normalized_doi}"
                ss_response = await self.client.get(ss_url, timeout=4.0)
                if ss_response.status_code == 200:
                    ss_data = ss_response.json()
                    abstract = ss_data.get("abstract", "")
                    if abstract:
                        return abstract
            except:
                pass
            return None
        
        # 并行请求Crossref和Semantic Scholar，第一个成功即返回（优化性能）
        try:
            # 创建任务
            crossref_task = asyncio.create_task(fetch_crossref())
            ss_task = asyncio.create_task(fetch_semantic_scholar())
            
            # 使用asyncio.wait，第一个完成且成功的即返回
            done, pending = await asyncio.wait(
                [crossref_task, ss_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 检查已完成的任务
            for task in done:
                try:
                    result = await task
                    if result:
                        # 取消其他任务
                        for p in pending:
                            p.cancel()
                            try:
                                await p
                            except asyncio.CancelledError:
                                pass
                        return result
                except Exception:
                    pass
            
            # 如果第一个完成的任务没有结果，等待另一个
            if pending:
                for task in pending:
                    try:
                        result = await task
                        if result:
                            return result
                    except Exception:
                        pass
        except Exception:
            pass
        
        return ""

    async def get_references_for_paper(self, paper: Dict) -> List[Dict]:
        """
        获取文献的参考文献列表
        
        Args:
            paper: 文献信息字典，应包含 openalex_id 或 doi
            
        Returns:
            参考文献列表，每个包含基本信息（title, authors, doi等）
        """
        references = []
        
        try:
            # 优先使用OpenAlex获取引用信息
            if paper.get("openalex_id") or paper.get("doi"):
                detail_url = None
                paper_identifier = None
                
                if paper.get("openalex_id"):
                    work_id = paper["openalex_id"]
                    detail_url = f"https://api.openalex.org/works/{work_id}"
                    paper_identifier = f"OpenAlex ID: {work_id}"
                elif paper.get("doi"):
                    # 标准化DOI格式
                    doi = self._normalize_doi(paper['doi'])
                    if doi:
                        detail_url = f"https://api.openalex.org/works/doi:{doi}"
                        paper_identifier = f"DOI: {doi}"
                    else:
                        print(f"DOI格式无效: {paper.get('doi')}")
                        return references
                else:
                    return references
                
                print(f"正在通过 {paper_identifier} 获取引用信息...")
                print(f"   URL: {detail_url}")
                
                detail_response = await self.client.get(detail_url, timeout=10.0)
                
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    
                    # 检查是否真的找到了文献
                    if not detail_data.get("id"):
                        print(f"OpenAlex未找到该文献")
                        return references
                    
                    print(f"成功获取文献信息: {detail_data.get('title', 'N/A')[:60]}...")
                    
                    referenced_works = detail_data.get("referenced_works", [])
                    
                    if not referenced_works:
                        print(f"该文献没有引用信息（referenced_works为空）")
                        return references
                    
                    print(f"找到 {len(referenced_works)} 个引用文献")
                    
                    # 批量获取引用的文献详情（优化并发）
                    semaphore = asyncio.Semaphore(10)  # 增加并发数从5到10
                    failed_count = 0
                    not_found_count = 0
                    
                    async def fetch_ref_info(ref_url: str):
                        nonlocal failed_count, not_found_count
                        async with semaphore:
                            try:
                                # 将OpenAlex Web URL转换为API URL
                                api_url = ref_url
                                if ref_url.startswith('https://openalex.org/'):
                                    # 格式: https://openalex.org/W4322580283
                                    work_id = ref_url.split('/')[-1]
                                    api_url = f"https://api.openalex.org/works/{work_id}"
                                elif ref_url.startswith('http://openalex.org/'):
                                    work_id = ref_url.split('/')[-1]
                                    api_url = f"https://api.openalex.org/works/{work_id}"
                                elif not ref_url.startswith('https://api.openalex.org/'):
                                    # 如果不是API URL，可能是OpenAlex ID，尝试构造API URL
                                    if ref_url.startswith('W'):
                                        # OpenAlex ID格式: W4322580283
                                        api_url = f"https://api.openalex.org/works/{ref_url}"
                                
                                ref_response = await self.client.get(api_url, timeout=5.0)
                                
                                if ref_response.status_code == 200:
                                    # 检查响应内容类型
                                    content_type = ref_response.headers.get('content-type', '').lower()
                                    if 'application/json' not in content_type:
                                        failed_count += 1
                                        return None
                                    
                                    try:
                                        ref_data = ref_response.json()
                                    except Exception as json_error:
                                        failed_count += 1
                                        return None
                                    
                                    # 检查返回的数据是否有效
                                    if not ref_data or not ref_data.get("id"):
                                        failed_count += 1
                                        return None
                                    
                                    # 提取摘要
                                    abstract = ""
                                    if ref_data.get("abstract"):
                                        abstract = self._extract_abstract_from_openalex_data(ref_data)
                                    
                                    # 如果OpenAlex没有摘要，尝试从其他数据源获取
                                    if not abstract:
                                        doi = ref_data.get("doi", "")
                                        if doi:
                                            abstract = await self._get_abstract_by_doi(doi)
                                    
                                    ref_paper = {
                                        "title": ref_data.get("title", ""),
                                        "authors": [a.get("author", {}).get("display_name", "") 
                                                  for a in ref_data.get("authorships", [])[:3]],
                                        "abstract": abstract,
                                        "doi": ref_data.get("doi", ""),
                                        "published": ref_data.get("publication_date", ""),
                                        "url": ref_data.get("primary_location", {}).get("landing_page_url") or ref_data.get("id", ""),
                                        "openalex_id": ref_data.get("id", "").split("/")[-1] if ref_data.get("id") else ""
                                    }
                                    return ref_paper
                                elif ref_response.status_code == 404:
                                    # 404是正常现象：某些引用文献在OpenAlex数据库中可能不存在或已被移除
                                    not_found_count += 1
                                    return None
                                else:
                                    failed_count += 1
                                    return None
                            except Exception as e:
                                failed_count += 1
                                return None
                    
                    ref_tasks = [fetch_ref_info(ref_url) for ref_url in referenced_works[:20]]  # 限制前20个
                    ref_results = await asyncio.gather(*ref_tasks, return_exceptions=True)
                    
                    for ref_result in ref_results:
                        if isinstance(ref_result, dict) and ref_result:
                            references.append(ref_result)
                        elif isinstance(ref_result, Exception):
                            failed_count += 1
                    
                    # 显示统计信息
                    success_count = len(references)
                    total_attempted = min(len(referenced_works), 20)
                    print(f"成功获取 {success_count} 篇引用文献的详细信息")
                    if not_found_count > 0 or failed_count > 0:
                        print(f"   统计: 尝试 {total_attempted} 个, 成功 {success_count} 个, 未找到 {not_found_count} 个, 失败 {failed_count} 个")
                        if not_found_count > 0:
                            print(f"   注: 部分引用文献在OpenAlex中不存在（404），这是正常现象")
                            print(f"       可能原因: 数据库不完整、文献已移除、ID变更等")
                else:
                    print(f"获取文献详情失败")
                    print(f"   状态码: {detail_response.status_code}")
                    print(f"   URL: {detail_url}")
                    
                    if detail_response.status_code == 404:
                        print(f"   原因: 文献未找到（可能是DOI不正确或OpenAlex中没有该文献）")
                    elif detail_response.status_code == 429:
                        print(f"   原因: API请求频率限制，请稍后再试")
                    else:
                        try:
                            error_text = detail_response.text[:200]
                            print(f"   错误信息: {error_text}")
                        except:
                            pass
        
        except Exception as e:
            print(f"获取参考文献列表错误: {e}")
            import traceback
            print(f"   详细错误:")
            traceback.print_exc()
        
        return references

    async def get_top_cited_references_from_papers(self, papers: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        从检索到的文献中提取所有引用文献，按引用量排序，返回前n个
        
        Args:
            papers: 文献列表（从搜索结果中获取）
            top_n: 返回引用量前n个的文献
            
        Returns:
            按引用量排序的引用文献列表（前n个），包含摘要等信息
        """
        print(f"\n开始提取引用文献，目标：找到引用量前 {top_n} 的文献...")
        
        # 1. 收集所有引用文献的标识符
        all_referenced_ids = set()
        referenced_mapping = {}  # 用于记录引用关系
        
        for paper in papers:
            referenced_works = paper.get('referenced_works', [])
            if referenced_works:
                for ref_id in referenced_works:
                    if isinstance(ref_id, str) and ref_id:
                        all_referenced_ids.add(ref_id)
                        # 记录哪些文献引用了这个引用文献
                        if ref_id not in referenced_mapping:
                            referenced_mapping[ref_id] = {
                                'ref_id': ref_id,
                                'cited_by_papers': []
                            }
                        referenced_mapping[ref_id]['cited_by_papers'].append(paper.get('title', 'Unknown'))
        
        if not all_referenced_ids:
            print("未找到引用文献")
            return []
        
        print(f"共找到 {len(all_referenced_ids)} 个唯一的引用文献")
        
        # 2. 批量获取引用文献的详细信息（包括引用量）
        referenced_papers = []
        semaphore = asyncio.Semaphore(15)  # 优化：增加并发数从10到15
        
        async def fetch_reference_info(ref_id: str):
            """获取单个引用文献的详细信息"""
            async with semaphore:
                try:
                    # 判断ref_id的类型并构造API URL
                    detail_url = ref_id
                    if ref_id.startswith('https://openalex.org/'):
                        # Web URL格式: https://openalex.org/W4322580283
                        work_id = ref_id.split('/')[-1]
                        detail_url = f"https://api.openalex.org/works/{work_id}"
                    elif ref_id.startswith('http://openalex.org/'):
                        work_id = ref_id.split('/')[-1]
                        detail_url = f"https://api.openalex.org/works/{work_id}"
                    elif ref_id.startswith('https://api.openalex.org/'):
                        # 已经是API URL，直接使用
                        detail_url = ref_id
                    elif ref_id.startswith('http'):
                        # 其他HTTP URL，可能需要检查是否为OpenAlex格式
                        if 'openalex.org' in ref_id and not 'api.openalex.org' in ref_id:
                            # 如果是openalex.org但不是api.openalex.org，转换为API URL
                            work_id = ref_id.split('/')[-1]
                            detail_url = f"https://api.openalex.org/works/{work_id}"
                        else:
                            # 其他URL，尝试直接使用
                            detail_url = ref_id
                    elif '/' in ref_id:
                        # 可能是 OpenAlex ID (格式: W4322580283)
                        if ref_id.startswith('W'):
                            detail_url = f"https://api.openalex.org/works/{ref_id}"
                        else:
                            detail_url = f"https://api.openalex.org/works/{ref_id}"
                    else:
                        # 可能是DOI，先标准化
                        normalized_doi = self._normalize_doi(ref_id)
                        if normalized_doi:
                            detail_url = f"https://api.openalex.org/works/doi:{normalized_doi}"
                        elif ref_id.startswith('W'):
                            # 可能是OpenAlex ID（没有斜杠）
                            detail_url = f"https://api.openalex.org/works/{ref_id}"
                        else:
                            # 如果标准化失败，尝试直接使用
                            detail_url = f"https://api.openalex.org/works/doi:{ref_id}"
                    
                    response = await self.client.get(detail_url, timeout=5.0)
                    if response.status_code == 200:
                        # 检查响应内容类型
                        content_type = response.headers.get('content-type', '').lower()
                        if 'application/json' not in content_type:
                            print(f"响应不是JSON格式，Content-Type: {content_type}, URL: {detail_url}")
                            return None
                        
                        try:
                            data = response.json()
                        except Exception as json_error:
                            print(f"JSON解析失败: {json_error}, URL: {detail_url}")
                            print(f"   响应内容（前200字符）: {response.text[:200]}")
                            return None
                        
                        # 检查返回的数据是否有效
                        if not data or not data.get("id"):
                            print(f"返回的数据无效: {detail_url}")
                            return None
                        
                        # 提取摘要
                        abstract = ""
                        if data.get("abstract"):
                            abstract = self._extract_abstract_from_openalex_data(data)
                        
                        # 如果OpenAlex没有摘要，尝试从其他数据源获取
                        if not abstract:
                            doi = data.get("doi", "")
                            if doi:
                                abstract = await self._get_abstract_by_doi(doi)
                        
                        # 获取引用量
                        cited_by_count = data.get("cited_by_count", 0)
                        
                        ref_paper = {
                            "title": data.get("title", ""),
                            "authors": [a.get("author", {}).get("display_name", "") 
                                      for a in data.get("authorships", [])[:5]],
                            "abstract": abstract,
                            "doi": data.get("doi", ""),
                            "published": data.get("publication_date", ""),
                            "cited_by_count": cited_by_count,  # 引用量
                            "url": data.get("primary_location", {}).get("landing_page_url") or data.get("id", ""),
                            "openalex_id": data.get("id", "").split("/")[-1] if data.get("id") else "",
                            "source": "OpenAlex (Cited Reference)",
                            "cited_by_papers": referenced_mapping.get(ref_id, {}).get('cited_by_papers', [])
                        }
                        
                        # 尝试获取PDF链接
                        for location in data.get("locations", []):
                            if location.get("pdf_url"):
                                ref_paper["pdf_url"] = location["pdf_url"]
                                break
                        
                        if not ref_paper.get("pdf_url"):
                            oa_info = data.get("open_access", {})
                            if oa_info.get("oa_url"):
                                ref_paper["pdf_url"] = oa_info["oa_url"]
                        
                        return ref_paper
                    elif response.status_code == 404:
                        # 404是正常现象，不打印错误信息（会在统计中显示）
                        return None
                    else:
                        # 其他错误也不打印，避免信息过多
                        return None
                except Exception as e:
                    # 静默处理异常，避免信息过多
                    return None
        
        # 批量获取引用文献信息
        print(f"正在批量获取 {len(all_referenced_ids)} 个引用文献的详细信息...")
        ref_tasks = [fetch_reference_info(ref_id) for ref_id in list(all_referenced_ids)]
        ref_results = await asyncio.gather(*ref_tasks, return_exceptions=True)
        
        # 统计信息
        success_count = 0
        not_found_count = 0
        failed_count = 0
        
        # 过滤掉失败的结果
        for result in ref_results:
            if isinstance(result, dict) and result:
                referenced_papers.append(result)
                success_count += 1
            elif isinstance(result, Exception):
                failed_count += 1
        
        # 计算404的数量（会在fetch_reference_info中记录，这里简化处理）
        total_attempted = len(all_referenced_ids)
        not_found_count = total_attempted - success_count - failed_count
        
        print(f"成功获取 {success_count} 个引用文献的详细信息")
        if not_found_count > 0 or failed_count > 0:
            print(f"   统计: 尝试 {total_attempted} 个, 成功 {success_count} 个, 未找到 {not_found_count} 个, 失败 {failed_count} 个")
            if not_found_count > 0:
                print(f"   注: 部分引用文献在OpenAlex中不存在（404），这是正常现象")
                print(f"       可能原因: 数据库不完整、文献已移除、ID变更、版权限制等")
        
        # 3. 按引用量排序
        referenced_papers.sort(key=lambda x: x.get('cited_by_count', 0), reverse=True)
        
        # 4. 返回前n个
        top_references = referenced_papers[:top_n]
        
        print(f"返回引用量前 {len(top_references)} 的文献:")
        for i, ref in enumerate(top_references[:5], 1):  # 只打印前5个
            print(f"  {i}. {ref.get('title', 'N/A')[:60]}... (引用量: {ref.get('cited_by_count', 0)})")
        
        return top_references

    async def search_and_extract(self, query: str, max_results: int = 10,
                                 extract_pdf: bool = True,
                                 sources: Optional[List[str]] = None,
                                 sort_by: str = "relevance",
                                 include_references: bool = False) -> List[Dict]:
        """
        搜索文献并提取内容
        
        Args:
            query: 搜索主题
            max_results: 每个数据源的最大结果数
            extract_pdf: 是否下载并提取PDF内容
            sources: 要使用的数据源列表，可选值: ['arxiv', 'openalex', 'crossref']
                    如果为None，则使用所有可用数据源
            sort_by: 排序方式，"relevance" 按相关性排序（默认），"date" 按时间排序，"citations" 按引用次数排序
            include_references: 是否获取引用文献列表，默认为False（可提高搜索速度）
            
        Returns:
            包含完整内容的文献列表
        """
        print(f"开始搜索主题: {query}, 排序方式: {sort_by}")

        if sources is None:
            sources = ['arxiv', 'openalex', 'crossref']
        
        sources = [s.lower() for s in sources]

        search_tasks = []
        source_names = []
        
        if 'arxiv' in sources:
            search_tasks.append(self.search_arxiv(query, max_results, sort_by=sort_by))
            source_names.append('arXiv')
        if 'openalex' in sources:
            search_tasks.append(self.search_openalex(query, max_results, sort_by=sort_by, include_references=include_references))
            source_names.append('OpenAlex')
        if 'crossref' in sources:
            search_tasks.append(self.search_crossref(query, max_results, sort_by=sort_by, include_references=include_references))
            source_names.append('Crossref')

        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        else:
            print("警告: 未选择任何数据源")
            return []

        all_papers = []
        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                print(f"{source_name}搜索异常: {result}")
            elif result:
                all_papers.extend(result)
                print(f"{source_name}找到 {len(result)} 篇文献")
        
        # 并行查找Unpaywall开放获取版本
        if 'crossref' in sources or 'unpaywall' in sources:
            papers_without_pdf = [p for p in all_papers if not p.get('pdf_url') and p.get('doi')]
            if papers_without_pdf:
                print(f"通过Unpaywall并行查找 {len(papers_without_pdf)} 篇文献的开放获取版本...")
                async def fetch_unpaywall(paper):
                    try:
                        unpaywall_url = f"https://api.unpaywall.org/v2/{paper['doi']}?email=openaccess@example.com"
                        unpaywall_response = await self.client.get(unpaywall_url, timeout=3.0)
                        if unpaywall_response.status_code == 200:
                            unpaywall_data = unpaywall_response.json()
                            if unpaywall_data.get("is_oa") and unpaywall_data.get("best_oa_location"):
                                paper["pdf_url"] = unpaywall_data["best_oa_location"].get("url_for_pdf") or \
                                                  unpaywall_data["best_oa_location"].get("url_for_landing_page")
                                if paper.get("source") == "Crossref":
                                    paper["source"] = "Crossref (via Unpaywall)"
                    except:
                        pass
                
                # 并行处理，限制并发数
                semaphore = asyncio.Semaphore(10)
                async def limited_fetch(paper):
                    async with semaphore:
                        return await fetch_unpaywall(paper)
                
                await asyncio.gather(*[limited_fetch(p) for p in papers_without_pdf], return_exceptions=True)

        if extract_pdf:
            print(f"开始提取PDF内容，共{len(all_papers)}篇文献")
            for paper in all_papers:
                if paper.get('pdf_url'):
                    print(f"下载PDF: {paper.get('title', '')[:50]}...")
                    pdf_content = await self.download_pdf(paper['pdf_url'])
                    if pdf_content:
                        pdf_text = self.extract_text_from_pdf(pdf_content)
                        paper['pdf_text'] = pdf_text
                        paper['pdf_size'] = len(pdf_content)
                    else:
                        paper['pdf_text'] = None
                        paper['pdf_size'] = 0
                else:
                    paper['pdf_text'] = None
                    paper['pdf_size'] = 0

        # 补充缺失的摘要：对于没有摘要但有DOI的文献，批量使用多源备用方案（优化：增加并发数）
        papers_without_abstract = [p for p in all_papers if not p.get('abstract') and p.get('doi')]
        if papers_without_abstract:
            print(f"为 {len(papers_without_abstract)} 篇文献批量补充摘要...")
            # 优化：使用多源备用方案，提高成功率
            semaphore = asyncio.Semaphore(10)  # 增加并发数从5到10
            async def fetch_abstract_multi_source(paper):
                async with semaphore:
                    doi = paper.get('doi')
                    if not doi:
                        return paper, ""
                    # 先尝试OpenAlex（快速）
                    abstract = await self._get_abstract_from_openalex_by_doi(doi)
                    if abstract:
                        return paper, abstract
                    # 如果OpenAlex没有，使用多源备用方案
                    abstract = await self._get_abstract_by_doi(doi)
                    return paper, abstract
            
            tasks = [fetch_abstract_multi_source(p) for p in papers_without_abstract]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = 0
            for result in results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    paper, abstract = result
                    if abstract:
                        paper['abstract'] = abstract
                        success_count += 1
            if success_count > 0:
                print(f"  ✓ 成功为 {success_count} 篇文献补充摘要")

        print(f"搜索完成，共找到{len(all_papers)}篇文献")
        return all_papers

    async def search_and_extract_with_top_references(self, query: str, max_results: int = 10,
                                                     extract_pdf: bool = True,
                                                     sources: Optional[List[str]] = None,
                                                     sort_by: str = "relevance",
                                                     extract_top_cited_refs: bool = True,
                                                     top_cited_n: int = 10) -> Dict:
        """
        搜索文献并提取内容，同时提取引用量前n的引用文献
        
        Args:
            query: 搜索主题
            max_results: 每个数据源的最大结果数
            extract_pdf: 是否下载并提取PDF内容
            sources: 要使用的数据源列表
            sort_by: 排序方式，"relevance" 按相关性排序（默认），"date" 按时间排序，"citations" 按引用次数排序
            extract_top_cited_refs: 是否提取引用量前n的引用文献
            top_cited_n: 返回引用量前n的引用文献数量
            
        Returns:
            包含搜索结果和顶级引用文献的字典：
            {
                "papers": List[Dict],  # 搜索结果
                "top_cited_references": List[Dict]  # 引用量前n的引用文献
            }
        """
        # 先进行常规搜索
        papers = await self.search_and_extract(
            query=query,
            max_results=max_results,
            extract_pdf=extract_pdf,
            sources=sources,
            sort_by=sort_by
        )
        
        result = {
            "papers": papers,
            "top_cited_references": []
        }
        
        # 如果启用了提取顶级引用文献功能
        if extract_top_cited_refs and papers:
            top_references = await self.get_top_cited_references_from_papers(papers, top_n=top_cited_n)
            result["top_cited_references"] = top_references
        
        return result

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()


# 便捷函数
async def search_literature(query: str, max_results: int = 10,
                           extract_pdf: bool = True,
                           sources: Optional[List[str]] = None) -> List[Dict]:
    """
    搜索文献并获取内容
    
    Args:
        query: 搜索主题
        max_results: 每个数据源的最大结果数
        extract_pdf: 是否提取PDF内容
        sources: 要使用的数据源列表，可选值: ['arxiv', 'openalex', 'crossref']
                如果为None，则使用所有可用数据源
        
    Returns:
        文献列表
    """
    searcher = LiteratureSearcher()
    try:
        results = await searcher.search_and_extract(query, max_results, extract_pdf, sources)
        return results
    finally:
        await searcher.close()


if __name__ == "__main__":
    async def basic_search_demo():
        """leaf area index"""
        query = "resnet"
        results = await search_literature(query, max_results=3, extract_pdf=False, sources=['resnet'])
        print(f"\n找到 {len(results)} 篇文献:")
        for i, paper in enumerate(results, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   来源: {paper['source']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}")
            print(f"   摘要: {paper['abstract']}")
    
    asyncio.run(basic_search_demo())

