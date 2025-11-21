from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class PaperResult:
    """
    统一的论文搜索结果结构体
    """
    title: str
    authors: List[str]
    abstract: Optional[str]
    year: Optional[int]  # 发表年份
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str = ""  # 数据源名称，如 "OpenAlex", "arXiv", "OpenReview"
    # 额外的源特定字段
    arxiv_id: Optional[str] = None
    category: Optional[str] = None
    venue: Optional[str] = None
    published_date: Optional[str] = None  # 完整日期字符串


class BaseSearcher(ABC):
    """
    文献搜索基类
    所有搜索器都应该继承此类并实现 search 方法
    """
    
    def __init__(self, source_name: str):
        """
        初始化搜索器
        
        Args:
            source_name: 数据源名称
        """
        self.source_name = source_name
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10, sorted: Optional[str] = None, **kwargs) -> List[PaperResult]:
        """
        搜索文献（异步方法）
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sorted: 排序方式，不同数据源支持的排序选项不同
            **kwargs: 其他搜索参数（如 venue, year 等）
            
        Returns:
            论文结果列表
        """
        pass
    
    def _normalize_authors(self, authors) -> List[str]:
        """
        标准化作者列表格式
        对于中文作者，将"名 姓"格式转换为"姓 名"格式（如"闻捷 范" -> "范闻捷"）
        
        Args:
            authors: 作者数据（可能是列表、字符串或其他格式）
            
        Returns:
            标准化的作者列表
        """
        if not authors:
            return []
        
        def normalize_chinese_author(author_str: str) -> str:
            """
            规范化单个中文作者名字
            将"名 姓"格式转换为"姓 名"格式
            """
            if not author_str or not isinstance(author_str, str):
                return author_str
            
            author_str = author_str.strip()
            
            # 检测是否包含中文字符
            if not re.search(r'[\u4e00-\u9fff]', author_str):
                return author_str
            
            # 检查是否是"名 姓"格式（两个中文字符，中间有空格）
            # 匹配模式：中文字符 + 空格 + 中文字符
            pattern = r'^([\u4e00-\u9fff]+)\s+([\u4e00-\u9fff]+)$'
            match = re.match(pattern, author_str)
            
            if match:
                # 找到匹配，将"名 姓"转换为"姓 名"
                given_name = match.group(1)  # 名
                surname = match.group(2)     # 姓
                return f"{surname}{given_name}"  # 返回"姓 名"格式（去掉空格）
            
            # 如果不匹配模式，保持原样
            return author_str
        
        if isinstance(authors, str):
            return [normalize_chinese_author(authors)]
        
        if isinstance(authors, list):
            return [normalize_chinese_author(str(a)) for a in authors if a]
        
        return []
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """
        从日期字符串中提取年份
        
        Args:
            date_str: 日期字符串，如 "2024-01-15" 或 "2024"
            
        Returns:
            年份整数，如果无法提取则返回 None
        """
        if not date_str:
            return None
        
        try:
            # 尝试提取年份（前4位数字）
            import re
            match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
            if match:
                return int(match.group())
        except:
            pass
        
        return None

