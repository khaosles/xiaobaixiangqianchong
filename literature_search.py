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
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        从arXiv搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            文献列表，每个包含title, authors, abstract, pdf_url等信息
        """
        try:
            url = "https://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
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
                    'url': None
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

    async def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        从PubMed搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            文献列表
        """
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "pub_date"
            }

            search_response = await self.client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()

            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return []

            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }

            fetch_response = await self.client.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()

            root = ET.fromstring(fetch_response.text)
            ns = {'': 'http://www.ncbi.nlm.nih.gov'}

            papers = []
            for article in root.findall('.//PubmedArticle'):
                try:
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ''

                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ''

                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")

                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''

                    pub_date_elem = article.find('.//PubDate')
                    year = ''
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        if year_elem is not None:
                            year = year_elem.text

                    paper = {
                        'source': 'PubMed',
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'published': year,
                        'pmid': pmid,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else None,
                        'pdf_url': None  
                    }
                    papers.append(paper)
                except Exception as e:
                    print(f"解析PubMed文章错误: {e}")
                    continue

            return papers

        except Exception as e:
            print(f"PubMed搜索错误: {e}")
            return []

    async def search_openalex(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        从OpenAlex搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            文献列表
        """
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per_page": min(max_results, 200),  # OpenAlex限制每页最多200
                "sort": "publication_date:desc"
            }
            
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
                    "openalex_id": work.get("id", "").split("/")[-1] if work.get("id") else ""
                }
                
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
            
            # 批量获取缺失的摘要
            if works_data:
                tasks = []
                for paper, work in works_data:
                    if work.get("doi"):
                        tasks.append(self._get_abstract_from_openalex_by_doi(work.get("doi")))
                    elif work.get("id"):
                        work_id = work.get("id").split("/")[-1] if "/" in work.get("id", "") else work.get("id")
                        tasks.append(self._get_abstract_from_openalex_by_work_id(work_id))
                    else:
                        tasks.append(asyncio.sleep(0))  # 占位
                
                if tasks:
                    abstracts = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, (paper, _) in enumerate(works_data):
                        if i < len(abstracts) and isinstance(abstracts[i], str) and abstracts[i]:
                            paper["abstract"] = abstracts[i]
            
            return papers
            
        except Exception as e:
            print(f"OpenAlex搜索错误: {e}")
            return []

    async def search_crossref(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        从Crossref搜索文献
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            文献列表
        """
        try:
            url = "https://api.crossref.org/works"
            params = {
                "query": query,
                "rows": min(max_results, 1000),  # Crossref允许最多1000
                "sort": "relevance",
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
                    "pdf_url": None
                }
                
                if paper["doi"]:
                    paper["url"] = f"https://doi.org/{paper['doi']}"
                
                papers.append(paper)
                # 如果没有摘要但有DOI，记录需要补充
                if not abstract and paper.get("doi"):
                    crossref_papers.append(paper)
            
            # 批量通过DOI从OpenAlex获取摘要
            if crossref_papers:
                tasks = [self._get_abstract_from_openalex_by_doi(p.get("doi")) for p in crossref_papers]
                abstracts = await asyncio.gather(*tasks, return_exceptions=True)
                for i, paper in enumerate(crossref_papers):
                    if i < len(abstracts) and isinstance(abstracts[i], str) and abstracts[i]:
                        paper["abstract"] = abstracts[i]
            
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
            openalex_response = await self.client.get(openalex_url, timeout=5.0)
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
            detail_response = await self.client.get(detail_url, timeout=5.0)
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

    async def search_and_extract(self, query: str, max_results: int = 10,
                                 extract_pdf: bool = True,
                                 sources: Optional[List[str]] = None) -> List[Dict]:
        """
        搜索文献并提取内容
        
        Args:
            query: 搜索主题
            max_results: 每个数据源的最大结果数
            extract_pdf: 是否下载并提取PDF内容
            sources: 要使用的数据源列表，可选值: ['arxiv', 'pubmed', 'openalex', 'crossref']
                    如果为None，则使用所有数据源
            
        Returns:
            包含完整内容的文献列表
        """
        print(f"开始搜索主题: {query}")

        if sources is None:
            sources = ['arxiv', 'pubmed', 'openalex', 'crossref']
        
        sources = [s.lower() for s in sources]

        search_tasks = []
        source_names = []
        
        if 'arxiv' in sources:
            search_tasks.append(self.search_arxiv(query, max_results))
            source_names.append('arXiv')
        if 'pubmed' in sources:
            search_tasks.append(self.search_pubmed(query, max_results))
            source_names.append('PubMed')
        if 'openalex' in sources:
            search_tasks.append(self.search_openalex(query, max_results))
            source_names.append('OpenAlex')
        if 'crossref' in sources:
            search_tasks.append(self.search_crossref(query, max_results))
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
        
        if 'crossref' in sources or 'unpaywall' in sources:
            print("通过Unpaywall查找开放获取版本...")
            for paper in all_papers:
                if not paper.get('pdf_url') and paper.get('doi'):
                    try:
                        unpaywall_url = f"https://api.unpaywall.org/v2/{paper['doi']}?email=openaccess@example.com"
                        unpaywall_response = await self.client.get(unpaywall_url, timeout=5.0)
                        if unpaywall_response.status_code == 200:
                            unpaywall_data = unpaywall_response.json()
                            if unpaywall_data.get("is_oa") and unpaywall_data.get("best_oa_location"):
                                paper["pdf_url"] = unpaywall_data["best_oa_location"].get("url_for_pdf") or \
                                                  unpaywall_data["best_oa_location"].get("url_for_landing_page")
                                if paper.get("source") == "Crossref":
                                    paper["source"] = "Crossref (via Unpaywall)"
                    except:
                        pass

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

        # 补充缺失的摘要：对于没有摘要但有DOI的文献，批量从OpenAlex获取
        papers_without_abstract = [p for p in all_papers if not p.get('abstract') and p.get('doi')]
        if papers_without_abstract:
            print(f"为 {len(papers_without_abstract)} 篇文献批量补充摘要...")
            tasks = [self._get_abstract_from_openalex_by_doi(p.get('doi')) for p in papers_without_abstract]
            abstracts = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = 0
            for i, paper in enumerate(papers_without_abstract):
                if i < len(abstracts) and isinstance(abstracts[i], str) and abstracts[i]:
                    paper['abstract'] = abstracts[i]
                    success_count += 1
            if success_count > 0:
                print(f"  ✓ 成功为 {success_count} 篇文献补充摘要")

        print(f"搜索完成，共找到{len(all_papers)}篇文献")
        return all_papers

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
        sources: 要使用的数据源列表，可选值: ['arxiv', 'pubmed', 'openalex', 'crossref']
                如果为None，则使用所有数据源
        
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
        query = "What are the latest advances in transformer models?"
        results = await search_literature(query, max_results=3, extract_pdf=False)
        print(f"\n找到 {len(results)} 篇文献:")
        for i, paper in enumerate(results, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   来源: {paper['source']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}")
            print(f"   摘要: {paper['abstract']}")
    
    asyncio.run(basic_search_demo())

