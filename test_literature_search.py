#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文献搜索功能测试脚本
"""
import asyncio
from literature_search import search_literature


async def main():
    """测试文献搜索功能"""
    # 测试主题
    query = "transformer models"
    print(f"正在搜索主题: {query}\n")
    print("=" * 80)
    
    # 搜索文献（不提取PDF内容，速度更快）
    results = await search_literature(
        query=query,
        max_results=5,
        extract_pdf=False  # 设置为True会下载PDF并提取文本，速度较慢
    )
    
    print(f"\n找到 {len(results)} 篇文献:\n")
    print("=" * 80)
    
    for i, paper in enumerate(results, 1):
        print(f"\n【文献 {i}】")
        print(f"标题: {paper['title']}")
        print(f"来源: {paper['source']}")
        print(f"作者: {', '.join(paper['authors'][:5])}")
        if paper.get('published'):
            print(f"发表时间: {paper['published']}")
        print(f"摘要: {paper['abstract'][:200]}...")
        if paper.get('url'):
            print(f"链接: {paper['url']}")
        if paper.get('pdf_url'):
            print(f"PDF: {paper['pdf_url']}")
        print("-" * 80)
    
    # 如果需要提取PDF内容，可以这样测试（注意：会比较慢）
    print("\n\n是否要测试PDF内容提取？(这需要下载PDF文件，可能较慢)")
    print("取消注释下面的代码来测试PDF提取功能：")
    print("""
    # results_with_pdf = await search_literature(
    #     query=query,
    #     max_results=2,  # 只测试2篇，因为下载PDF较慢
    #     extract_pdf=True
    # )
    # 
    # for paper in results_with_pdf:
    #     if paper.get('pdf_text'):
    #         print(f"\\n论文: {paper['title']}")
    #         print(f"PDF文本长度: {len(paper['pdf_text'])} 字符")
    #         print(f"PDF文本预览: {paper['pdf_text'][:500]}...")
    """)


if __name__ == "__main__":
    asyncio.run(main())

