import os
import json
import re
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from literature_search import LiteratureSearcher
from pdf_processor import extract_text_from_pdf_base64
from rag_utils import SimpleRAG


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
    
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计总字符数量（排除空格和标点）
    total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
    
    if total_chars == 0:
        return 'en'
    
    # 如果中文字符占比超过30%，认为是中文
    if chinese_chars / total_chars > 0.3:
        return 'zh'
    
    return 'en'

load_dotenv()

app = FastAPI(title="AI Scientist Challenge Submission")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

reasoning_client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)


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

        print(f"[literature_review] Received query: {query}")
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        # 检测用户输入的语言
        language = detect_language(query)
        print(f"[literature_review] Detected language: {language}")

        async def generate():
            try:
                # 步骤1: 搜索相关文献以增强上下文
                papers_context = ""
                papers_references = []  # 用于存储参考文献信息
                try:
                    searcher = LiteratureSearcher()
                    try:
                        papers = await searcher.search_and_extract(
                            query=query,
                            max_results=15,
                            extract_pdf=False,
                            sources=['openalex', 'crossref']
                        )
                        
                        # 构建论文摘要信息（只包含有摘要的文献，根据语言调整格式）
                        if papers:
                            papers_list = []
                            ref_index = 1
                            for paper in papers:
                                # 只处理有摘要的文献
                                if not paper.get('abstract') or not paper.get('abstract').strip():
                                    continue
                                
                                # 构建论文信息
                                if language == 'zh':
                                    paper_info = f"[{ref_index}] {paper.get('title', '未知')}"
                                    authors_str = ""
                                    if paper.get('authors'):
                                        authors_str = ', '.join(paper.get('authors', [])[:3])
                                        paper_info += f" - {authors_str}"
                                    if paper.get('published'):
                                        paper_info += f" ({paper.get('published')})"
                                    paper_info += f"\n   摘要: {paper.get('abstract', '')[:300]}"
                                else:
                                    paper_info = f"[{ref_index}] {paper.get('title', 'Unknown')}"
                                    authors_str = ""
                                    if paper.get('authors'):
                                        authors_str = ', '.join(paper.get('authors', [])[:3])
                                        paper_info += f" - {authors_str}"
                                    if paper.get('published'):
                                        paper_info += f" ({paper.get('published')})"
                                    paper_info += f"\n   Abstract: {paper.get('abstract', '')[:300]}"
                                
                                papers_list.append(paper_info)
                                
                                # 保存参考文献信息用于后续引用
                                ref_info = {
                                    'index': ref_index,
                                    'title': paper.get('title', ''),
                                    'authors': paper.get('authors', [])[:3] if paper.get('authors') else [],
                                    'published': paper.get('published', ''),
                                    'url': paper.get('url', ''),
                                    'doi': paper.get('doi', '')
                                }
                                papers_references.append(ref_info)
                                
                                ref_index += 1
                                if ref_index > 10:  # 最多10篇文献
                                    break
                            
                            if papers_list:
                                if language == 'zh':
                                    papers_context = "\n\n相关文献:\n" + "\n\n".join(papers_list)
                                else:
                                    papers_context = "\n\nRelevant Papers:\n" + "\n\n".join(papers_list)
                    finally:
                        await searcher.close()
                except Exception as e:
                    print(f"[literature_review] 文献搜索警告: {e}，继续生成综述")
                    # 即使搜索失败也继续生成综述

                # 步骤2: 根据语言准备prompt进行文献综述
                if language == 'zh':
                    citation_instruction = """
在综述中，请使用 [1], [2], [3] 等格式引用上述文献。在适当的地方插入引用，例如："根据研究[1]，transformer模型在自然语言处理中取得了显著进展。"
"""
                    if not papers_context:
                        citation_instruction = ""
                    
                    prompt = f"""请对以下研究主题进行全面的文献综述：

{query}

{papers_context}

{citation_instruction}

请提供一份结构化的文献综述，使用Markdown格式，包含以下部分：
- 研究背景：该研究领域的历史背景和重要性
- 关键主题：主要研究主题和核心概念
- 研究趋势：最新发展和新兴技术
- 研究空白：局限性、未解决的问题和未来研究方向

{citation_instruction}

请确保综述全面、准确、有深度，并在适当的地方使用文献引用 [1], [2], [3] 等格式。"""
                else:
                    citation_instruction = """
In your review, please cite the above papers using [1], [2], [3] format. Insert citations at appropriate places, for example: "According to research [1], transformer models have achieved significant progress in natural language processing."
"""
                    if not papers_context:
                        citation_instruction = ""
                    
                    prompt = f"""Conduct a comprehensive literature review on the following topic:

{query}

{papers_context}

{citation_instruction}

Please provide a structured literature review in Markdown format covering:
- Background: Historical context and importance of the research area
- Key Themes: Main research themes and core concepts
- Current Trends: Recent developments and emerging technologies
- Research Gaps: Limitations, unsolved problems, and future directions

{citation_instruction}

Ensure the review is thorough, accurate, and insightful, and use citations [1], [2], [3] etc. at appropriate places."""

                # 调用LLM模型进行流式生成
                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                # 流式返回结果（使用JSON格式）
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

                # 在综述末尾添加参考文献列表
                if papers_references:
                    if language == 'zh':
                        references_text = "\n\n## 参考文献\n\n"
                    else:
                        references_text = "\n\n## References\n\n"
                    
                    for ref in papers_references:
                        if language == 'zh':
                            authors_str = ', '.join(ref['authors']) if ref['authors'] else '未知作者'
                            ref_line = f"[{ref['index']}] {ref['title']}. {authors_str}"
                            if ref['published']:
                                ref_line += f" ({ref['published']})"
                            if ref['doi']:
                                ref_line += f". DOI: {ref['doi']}"
                            elif ref['url']:
                                ref_line += f". URL: {ref['url']}"
                        else:
                            authors_str = ', '.join(ref['authors']) if ref['authors'] else 'Unknown Authors'
                            ref_line = f"[{ref['index']}] {ref['title']}. {authors_str}"
                            if ref['published']:
                                ref_line += f" ({ref['published']})"
                            if ref['doi']:
                                ref_line += f". DOI: {ref['doi']}"
                            elif ref['url']:
                                ref_line += f". URL: {ref['url']}"
                        
                        references_text += ref_line + "\n\n"
                    
                    # 发送参考文献
                    response_data = {
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {
                                "content": references_text
                            }
                        }]
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"

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
    PDF论文问答端点
    
    Input: {"query": "...", "pdf_content": "base64 string"}
    Output: SSE stream of Markdown
    
    使用RAG技术从PDF中提取相关信息并回答问题
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

        print(f"[paper_qa] 查询: {query[:50]}...")

        # 检测用户输入的语言
        language = detect_language(query)
        print(f"[paper_qa] Detected language: {language}")

        async def generate():
            try:
                # 提取PDF文本
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    yield f"data: {json.dumps({'object': 'error', 'message': error_msg})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # 使用RAG检索相关段落
                rag = SimpleRAG()
                relevant_chunks = await rag.retrieve_relevant_chunks(query, pdf_text, top_k=5)
                context = "\n\n".join(relevant_chunks)
                
                # 根据语言构建prompt
                if language == 'zh':
                    prompt = f"""你是一位学术论文分析专家。请基于以下PDF文档内容回答用户的问题。

用户问题: {query}

相关文档内容:
{context}

请基于文档内容准确回答问题。如果文档中没有相关信息，请说明。使用Markdown格式输出答案。"""
                else:
                    prompt = f"""You are an academic paper analysis expert. Please answer the user's question based on the following PDF document content.

User Question: {query}

Relevant Document Content:
{context}

Please answer accurately based on the document content. If there is no relevant information in the document, please state so. Output the answer in Markdown format."""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.2,
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
                print(f"[paper_qa] 错误: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
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


@app.post("/ideation")
async def ideation(request: Request):
    """
    研究想法生成端点
    
    Input: {"query": "..."}
    Output: SSE stream of Markdown
    
    使用deepseek-reasoner生成新颖的研究想法
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[ideation] 查询: {query}")

        # 检测用户输入的语言
        language = detect_language(query)
        print(f"[ideation] Detected language: {language}")

        async def generate():
            try:
                # 根据语言构建prompt
                if language == 'zh':
                    prompt = f"""你是一位富有创造力的研究专家。请基于以下研究领域或问题，生成一个新颖的研究想法。

研究领域/问题: {query}

请生成一个创新的研究想法，使用Markdown格式，包含以下部分：

## 问题
清晰描述当前存在的问题或挑战。

## 研究想法
提出一个新颖的研究想法或解决方案。

## 可行性
分析该研究想法的技术可行性和实施难度。

## 影响
评估该研究想法的潜在影响和意义。

请确保想法具有创新性、可行性和重要性。使用Markdown格式输出。"""
                else:
                    prompt = f"""You are a creative research expert. Please generate a novel research idea based on the following research area or problem.

Research Area/Problem: {query}

Please generate an innovative research idea in Markdown format, including the following sections:

## Problem
Clearly describe the current problems or challenges.

## Idea
Propose a novel research idea or solution.

## Feasibility
Analyze the technical feasibility and implementation difficulty of this research idea.

## Impact
Evaluate the potential impact and significance of this research idea.

Ensure the idea is innovative, feasible, and significant. Output in Markdown format."""

                stream = await reasoning_client.chat.completions.create(
                    model=os.getenv("SCI_LLM_REASONING_MODEL", "deepseek-reasoner"),
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
                
            except asyncio.TimeoutError:
                error_msg = "请求超时（10分钟限制）" if language == 'zh' else "Request timeout (10 minutes limit)"
                error_data = {
                    "object": "error",
                    "message": error_msg
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[ideation] 错误: {e}")
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


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    论文评审端点
    
    Input: {"query": "...", "pdf_content": "base64 string"}
    Output: SSE stream of Markdown
    
    必须包含以下部分：
    - Summary
    - Strengths
    - Weaknesses / Concerns
    - Questions for Authors
    - Score (Overall, Novelty, Technical Quality, Clarity, Confidence)
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_review] 开始评审论文...")

        # 检测用户输入的语言（如果有query则检测query，否则默认英文）
        language = detect_language(query) if query else 'en'
        print(f"[paper_review] Detected language: {language}")

        async def generate():
            try:
                # 提取PDF文本
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    yield f"data: {json.dumps({'object': 'error', 'message': error_msg})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # 限制PDF文本长度
                if len(pdf_text) > 15000:
                    truncate_msg = "\n\n[文本已截断...]" if language == 'zh' else "\n\n[Text truncated...]"
                    pdf_text = pdf_text[:15000] + truncate_msg
                
                # 根据语言构建评审prompt
                if language == 'zh':
                    review_query = query if query else "请对这篇论文进行全面评审"
                    prompt = f"""你是一位资深的学术论文评审专家。请对以下论文进行全面、客观的评审。

评审重点: {review_query}

论文内容:
{pdf_text[:15000]}

请生成详细的论文评审报告，使用Markdown格式，必须包含以下部分：

## 摘要
简要总结论文的主要内容、研究方法和主要发现。

## 优点
列出论文的主要优点和贡献。

## 缺点/关注点
指出论文的不足之处、存在的问题或需要改进的地方。

## 给作者的问题
提出3-5个关键问题，帮助作者改进论文。

## 评分
请对论文进行评分（每项满分10分，置信度满分5分）：
- 总体评分: X/10
- 新颖性: X/10
- 技术质量: X/10
- 清晰度: X/10
- 置信度: X/5

请确保评审客观、专业、有建设性。使用Markdown格式输出。"""
                else:
                    review_query = query if query else "Please conduct a comprehensive review of this paper"
                    prompt = f"""You are a senior academic paper review expert. Please conduct a comprehensive and objective review of the following paper.

Review Focus: {review_query}

Paper Content:
{pdf_text[:15000]}

Please generate a detailed paper review report in Markdown format, which must include the following sections:

## Summary
Briefly summarize the main content, research methods, and key findings of the paper.

## Strengths
List the main advantages and contributions of the paper.

## Weaknesses / Concerns
Point out the shortcomings, problems, or areas that need improvement in the paper.

## Questions for Authors
Propose 3-5 key questions to help authors improve the paper.

## Score
Please score the paper (each item out of 10 points, Confidence out of 5 points):
- Overall: X/10
- Novelty: X/10
- Technical Quality: X/10
- Clarity: X/10
- Confidence: X/5

Ensure the review is objective, professional, and constructive. Output in Markdown format."""

                stream = await reasoning_client.chat.completions.create(
                    model=os.getenv("SCI_LLM_REASONING_MODEL", "deepseek-reasoner"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            yield f"data: {delta_content}\n\n"

                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"[paper_review] 错误: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
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
    """返回测试页面"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
