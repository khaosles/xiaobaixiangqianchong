"""
PDF处理模块
支持多种PDF解析库
"""
import base64
from io import BytesIO
from typing import Optional
import PyPDF2
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


def extract_text_from_pdf_base64(pdf_base64: str) -> Optional[str]:
    """
    从base64编码的PDF中提取文本
    
    Args:
        pdf_base64: base64编码的PDF字符串
        
    Returns:
        提取的文本内容，失败返回None
    """
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        return extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        print(f"PDF base64解码错误: {e}")
        return None


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """
    从PDF字节内容中提取文本
    
    Args:
        pdf_bytes: PDF文件的字节内容
        
    Returns:
        提取的文本内容，失败返回None
    """
    # 优先使用PyMuPDF（更快更准确）
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text = page.get_text()
                if text:
                    text_parts.append(text)
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"PyMuPDF提取错误: {e}，尝试其他方法")
    
    # 备选：使用pdfplumber
    if PDFPLUMBER_AVAILABLE:
        try:
            pdf_file = BytesIO(pdf_bytes)
            text_parts = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"pdfplumber提取错误: {e}，尝试PyPDF2")
    
    # 最后备选：使用PyPDF2
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"PyPDF2提取错误: {e}")
        return None

