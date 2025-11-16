import os
import json
import re
import asyncio
from typing import List, Dict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from literature_search import LiteratureSearcher
from pdf_processor import extract_text_from_pdf_base64
from rag_utils import SimpleRAG
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

    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))

    if total_chars == 0:
        return 'en'

    if chinese_chars / total_chars > 0.3:
        return 'zh'

    return 'en'


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

        language = detect_language(query)
        print(f"[literature_review] Detected language: {language}")

        async def generate():
            try:
                papers_context = ""
                papers_references = []
                try:
                    searcher = LiteratureSearcher()
                    try:
                        latest_papers_task = searcher.search_and_extract(
                            query=query,
                            max_results=7,
                            extract_pdf=False,
                            sort_by='date'
                        )

                        cited_papers_task = searcher.search_and_extract(
                            query=query,
                            max_results=7,
                            extract_pdf=False,
                            sort_by='citations',
                            sources=['openalex', 'crossref']
                        )

                        latest_papers, cited_papers = await asyncio.gather(
                            latest_papers_task,
                            cited_papers_task,
                            return_exceptions=True
                        )

                        if isinstance(latest_papers, Exception):
                            print(f"[literature_review] 最新文献搜索错误: {latest_papers}")
                            latest_papers = []
                        if isinstance(cited_papers, Exception):
                            print(f"[literature_review] 高引用文献搜索错误: {cited_papers}")
                            cited_papers = []

                        latest_papers_filtered = [
                            paper for paper in latest_papers
                            if paper.get('abstract') and paper.get('abstract').strip()
                        ]
                        cited_papers_filtered = [
                            paper for paper in cited_papers
                            if paper.get('abstract') and paper.get('abstract').strip()
                        ]

                        all_papers = []
                        seen_titles = set()

                        for paper in cited_papers_filtered:
                            title = paper.get('title', '').lower().strip()
                            if title and title not in seen_titles:
                                all_papers.append(paper)
                                seen_titles.add(title)
                        
                        for paper in latest_papers_filtered:
                            title = paper.get('title', '').lower().strip()
                            if title and title not in seen_titles:
                                all_papers.append(paper)
                                seen_titles.add(title)

                        papers = all_papers
                        
                        cited_papers_set = {id(p) for p in cited_papers_filtered}
                        cited_count = sum(1 for p in papers if id(p) in cited_papers_set)
                        print(f"[literature_review] 最终文献列表包含 {cited_count} 篇高引用文献")

                        print(
                            f"[literature_review] 找到 {len(latest_papers)} 篇最新文献（过滤后 {len(latest_papers_filtered)} 篇）和 {len(cited_papers)} 篇高引用文献（过滤后 {len(cited_papers_filtered)} 篇），合并去重后共 {len(papers)} 篇")

                        if papers:
                            papers_list = []
                            for idx, paper in enumerate(papers, 1):
                                authors = paper.get('authors', [])[:3] if paper.get('authors') else []
                                published = paper.get('published', '')
                                year = ''
                                if published:
                                    published_str = str(published) if not isinstance(published, str) else published
                                    year_match = re.search(r'(\d{4})', published_str)
                                    if year_match:
                                        year = year_match.group(1)

                                author_surname = ''
                                author_display = ''
                                if authors:
                                    first_author = authors[0]
                                    if ',' in first_author:
                                        author_surname = first_author.split(',')[0].strip()
                                    else:
                                        author_surname = first_author.split()[-1]

                                    if len(authors) == 1:
                                        author_display = author_surname
                                    elif len(authors) == 2:
                                        second_author_parts = authors[1].split()
                                        second_surname = second_author_parts[-1] if len(second_author_parts) > 0 else \
                                            authors[1]
                                        author_display = f"{author_surname} and {second_surname}"
                                    else:
                                        author_display = f"{author_surname} et al."
                                else:
                                    author_surname = 'Unknown'
                                    author_display = 'Unknown'

                                title = paper.get('title', '未知' if language == 'zh' else 'Unknown')
                                abstract = paper.get('abstract', '')[:500]
                                url = paper.get('url', '')
                                doi = paper.get('doi', '')
                                citation_url = f"https://doi.org/{doi}" if doi else url

                                if language == 'zh':
                                    paper_info = f"{idx}. 标题: {title}\n   作者: {author_display}"
                                    if year:
                                        paper_info += f"\n   年份: {year}"
                                    paper_info += f"\n   摘要: {abstract}"
                                    if citation_url:
                                        paper_info += f"\n   URL: {citation_url}"
                                else:
                                    paper_info = f"{idx}. Title: {title}\n   Authors: {author_display}"
                                    if year:
                                        paper_info += f"\n   Year: {year}"
                                    paper_info += f"\n   Abstract: {abstract}"
                                    if citation_url:
                                        paper_info += f"\n   URL: {citation_url}"

                                papers_list.append(paper_info)

                                ref_info = {
                                    'author_surname': author_surname,
                                    'author_display': author_display,
                                    'title': title,
                                    'authors': authors,
                                    'published': published,
                                    'year': year,
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
                    finally:
                        await searcher.close()
                except Exception as e:
                    print(f"[literature_review] 文献搜索警告: {e}，继续生成综述")

                if language == 'zh':
                    if papers_context:
                        citation_rules = """
**引用规则（GB/T 7714格式）:**

- **必须尽可能使用提供的全部文献。请仔细阅读所有提供的文献，并在综述中充分引用它们。**
- 只能引用下面列出的文献。不要编造标题、作者、年份或研究成果。
- **严禁引入没有给定的文献。所有引用必须来自提供的文献列表。**
- 引用时可以使用两种格式：
  1. 括号内引用格式：(作者, 年份) 或 (作者等, 年份)，例如："transformer模型在自然语言处理中取得了显著进展 (Chen等, 2024)。"
  2. 作者在前格式：作者 (年份) 或 作者等 (年份)，例如："Chen等 (2024) 提出了..." 或 "根据 Chen (2024) 的研究..."
- 在综述末尾，必须包含一个 **参考文献** 部分，按出现顺序列出所有引用的文献，严格使用GB/T 7714格式：
  
  示例格式：
  CHEN X, WANG Y, ZHANG Z, et al. Diffusion Language Models Are Versatile Few-shot Learners[J/OL]. Journal Name, 2024. DOI: 10.xxxx/xxxx
  
  或：
  CHEN X, WANG Y. Latent Diffusion for Text Generation[J/OL]. Conference Name, 2023. https://aclanthology.org/2023.xxx.pdf

  GB/T 7714格式说明：
  - 作者格式：姓在前（大写），名首字母缩写，如 CHEN X, WANG Y
  - 多作者用逗号分隔，最后一个作者前用"，"连接
  - 超过3个作者时，列出前3个作者，然后加"，et al."
  - 文献类型标识：[J/OL]表示期刊/在线
  - 格式：作者. 标题[J/OL]. 期刊名/会议名, 年份. DOI或URL

- 如果文献在正文中没有被引用，不要将其包含在参考文献中。
- 所有观点必须严格基于提供的摘要。
- **如果给定的文献列表为空，则跳过参考文献部分，不要生成任何参考文献。**
"""
                    else:
                        citation_rules = ""

                    prompt = f"""请对以下研究主题进行全面的文献综述：

{query}

{papers_context}

{citation_rules}

请提供一份结构化的文献综述，使用Markdown格式，包含以下部分：

**重要提示：必须尽可能使用提供的全部文献。在撰写每个部分时，必须在适当的地方引用提供的文献。每个主要观点、研究发现、技术方法都应该有相应的文献引用支持。请确保在综述中引用了大部分或全部提供的文献。**

1. **研究背景**
   - 该研究领域的历史发展和演进过程
   - 该领域的重要性和现实意义
   - 研究问题产生的背景和动机

2. **关键主题**
   - 该领域的核心研究主题和概念框架
   - 主要理论观点和方法论
   - 关键技术的原理和应用

3. **研究现状**
   - 当前研究的主要进展和突破
   - 不同研究方向的发展状况
   - 重要研究成果和发现

4. **研究趋势**
   - 最新发展和前沿技术
   - 新兴的研究方向和方法
   - 技术演进趋势

5. **研究空白**
   - 当前研究的局限性
   - 尚未解决的问题和挑战
   - 潜在的改进方向

6. **结论**
   - 总结该领域的研究现状和主要贡献
   - 概述关键研究发现和进展
   - 指出未来研究的重要方向和机会

{citation_rules}

写作要求：
- 确保综述全面、准确、有深度，体现对领域的深入理解
- 写作时要写得全面一点，内容丰富一些，深入分析每个主题，提供详细的论述和解释
- **必须严格遵循引用规则，在每个部分都要在适当的地方引用文献，不能遗漏引用**
- **每个主要观点、研究发现、技术方法都必须有相应的文献引用支持**
- 在适当的地方使用作者年份格式的文献引用，格式为：(作者, 年份) 或 作者 (年份)
- 保持逻辑清晰，各部分之间衔接自然
- 不要输出多余的空白行，保持格式紧凑
- 在综述末尾自动生成参考文献部分，只包含实际引用的文献， 如果给定的文献列表为空，则跳过该步骤"""
                else:
                    if papers_context:
                        citation_rules = """
**Rules (GB/T 7714 format):**

- **You MUST use as many of the provided papers as possible. Please carefully read all provided papers and cite them extensively in your review.**
- Cite only papers listed below. DO NOT invent titles, authors, years, or findings.
- **DO NOT introduce any papers that are not provided. All citations must come from the provided paper list.**
- When citing, you can use two formats:
  1. Parenthetical citation: (Author, Year) or (Author et al., Year), for example: "transformer models have achieved significant progress in natural language processing (Chen et al., 2024)."
  2. Narrative citation: Author (Year) or Author et al. (Year), for example: "Chen et al. (2024) proposed..." or "According to Chen (2024)..."
- At the end, you MUST include a **References** section listing all cited papers in order of appearance, strictly using GB/T 7714 format:

  Example format:
  CHEN X, WANG Y, ZHANG Z, et al. Diffusion Language Models Are Versatile Few-shot Learners[J/OL]. Journal Name, 2024. DOI: 10.xxxx/xxxx
  
  or:
  CHEN X, WANG Y. Latent Diffusion for Text Generation[J/OL]. Conference Name, 2023. https://aclanthology.org/2023.xxx.pdf

  GB/T 7714 format notes:
  - Author format: Surname FIRST (uppercase), given name initials, e.g., CHEN X, WANG Y
  - Multiple authors separated by commas, last author connected with ", "
  - More than 3 authors: list first 3 authors, then add ", et al."
  - Document type identifier: [J/OL] means Journal/Online
  - Format: Authors. Title[J/OL]. Journal/Conference Name, Year. DOI or URL

- If a paper is not cited in the text, do NOT include it in References.
- Base all claims strictly on the abstracts provided.
- **If the provided papers list is empty, skip the References section entirely and do NOT generate any references.**
"""
                    else:
                        citation_rules = ""

                    prompt = f"""Conduct a comprehensive literature review on the following topic:

{query}

{papers_context}

{citation_rules}

Please provide a structured literature review in Markdown format covering:

**IMPORTANT: You MUST use as many of the provided papers as possible. When writing each section, you MUST cite the provided papers at appropriate places. Every major point, research finding, or technical method should be supported by relevant citations. Please ensure that you cite most or all of the provided papers in your review.**

1. **Background**
   - Historical development and evolution of the research field
   - Significance and real-world importance of the area
   - Context and motivation for the research questions

2. **Key Themes**
   - Core research themes and conceptual frameworks
   - Main theoretical perspectives and methodologies
   - Principles and applications of key technologies

3. **Current State**
   - Major advances and breakthroughs in current research
   - Development status of different research directions
   - Important research outcomes and findings

4. **Research Trends**
   - Latest developments and cutting-edge technologies
   - Emerging research directions and methods
   - Technological evolution trends

5. **Research Gaps**
   - Limitations of current research
   - Unsolved problems and challenges
   - Potential areas for improvement

6. **Conclusion**
   - Summarize the current state of research and major contributions in the field
   - Outline key research findings and progress
   - Identify important future research directions and opportunities

{citation_rules}

Writing Requirements:
- Ensure the review is thorough, accurate, and insightful, demonstrating deep understanding of the field
- Write comprehensively with rich content, provide detailed analysis for each topic, and offer thorough explanations and discussions
- **MUST strictly follow citation rules: cite papers at appropriate places in EVERY section, do NOT omit citations**
- **Every major point, research finding, or technical method MUST be supported by relevant citations**
- Use author-year format citations at appropriate places, in formats: (Author, Year) or Author (Year)
- Maintain clear logic and natural transitions between sections
- Do NOT output excessive blank lines, keep the format compact
- Automatically generate a References section at the end, including only papers actually cited in the text. If the provided papers list is empty, skip this step"""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                full_text = ""

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
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
                    print(f"[literature_review] 开始检测引用，共有 {len(papers_references)} 篇参考文献")

                    cited_refs = []

                    if full_text:
                        print(f"[literature_review] 生成文本长度: {len(full_text)} 字符")

                        for ref in papers_references:
                            author_surname = ref.get('author_surname', 'Unknown')
                            year = ref.get('year', '')
                            multiple_authors = ref.get('multiple_authors', False)
                            title = ref.get('title', '').lower()

                            patterns = []
                            if year:
                                year_str = str(year) if not isinstance(year, str) else year
                                if multiple_authors:
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?,? {year_str})")
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?, {year_str})")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? \\({year_str}\\)")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? {year_str}")
                                    patterns.append(f"（{re.escape(author_surname)} et al\\.?,? {year_str}）")
                                    patterns.append(f"（{re.escape(author_surname)}等,? {year_str}）")
                                else:
                                    patterns.append(f"({re.escape(author_surname)},? {year_str})")
                                    patterns.append(f"({re.escape(author_surname)} {year_str})")
                                    patterns.append(f"{re.escape(author_surname)} \\({year_str}\\)")
                                    patterns.append(f"{re.escape(author_surname)} {year_str}")
                                    patterns.append(f"（{re.escape(author_surname)},? {year_str}）")
                                    patterns.append(f"（{re.escape(author_surname)} {year_str}）")
                            else:
                                if multiple_authors:
                                    patterns.append(f"{re.escape(author_surname)} et al\\.")
                                    patterns.append(f"{re.escape(author_surname)}等")
                                else:
                                    patterns.append(re.escape(author_surname))

                            cited = False
                            for pattern in patterns:
                                if re.search(pattern, full_text, re.IGNORECASE):
                                    cited = True
                                    print(f"[literature_review] 找到引用: {author_surname} ({year})")
                                    break

                            if not cited and title:
                                if any('\u4e00' <= char <= '\u9fff' for char in title):
                                    title_keywords = title[:10] if len(title) >= 10 else title
                                    if len(title_keywords) > 3 and title_keywords in full_text.lower():
                                        cited = True
                                        print(f"[literature_review] 通过标题关键词找到引用: {author_surname} ({year})")
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
                                                print(
                                                    f"[literature_review] 通过标题关键词找到引用: {author_surname} ({year})")

                            if cited:
                                cited_refs.append(ref)
                    else:
                        print(f"[literature_review] 警告: full_text 为空，无法检测引用")
                        cited_refs = papers_references

                    print(f"[literature_review] 检测到 {len(cited_refs)} 篇被引用的文献")

                    if len(cited_refs) < len(papers_references) * 0.5 and len(papers_references) > 0:
                        print(
                            f"[literature_review] 警告: 检测到的引用数量较少（{len(cited_refs)}/{len(papers_references)}），可能是匹配模式问题")
                        print(f"[literature_review] 将包含所有提供的文献以确保完整性")
                        cited_refs = papers_references
                    elif len(cited_refs) == 0 and len(papers_references) > 0:
                        print(f"[literature_review] 警告: 未检测到任何引用，将包含所有提供的文献")
                        cited_refs = papers_references

                    if cited_refs:
                        print(f"[literature_review] 模型应已自动生成参考文献，检测到 {len(cited_refs)} 篇被引用的文献")
                    else:
                        print(f"[literature_review] 警告: 未找到任何被引用的文献")
                else:
                    print(f"[literature_review] 警告: papers_references 为空")

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

        print(f"[paper_qa] Received query: {query}")
        print(f"[paper_qa] Using reasoning model: {os.getenv('SCI_LLM_REASONING_MODEL')}")

        language = detect_language(query)
        print(f"[paper_qa] Detected language: {language}")

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
                        if reasoning_content:
                            print(f"[paper_qa] Reasoning: {reasoning_content}", flush=True)

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

        print(f"[ideation] Received query: {query}")
        print(f"[ideation] Using {len(reference_ideas)} hardcoded reference ideas for embedding similarity")
        print(f"[ideation] Using LLM model: {os.getenv('SCI_LLM_MODEL')}")
        print(f"[ideation] Using embedding model: {os.getenv('SCI_EMBEDDING_MODEL')}")

        language = detect_language(query)
        print(f"[ideation] Detected language: {language}")

        async def generate():
            try:
                if language == 'zh':
                    prompt = f"""为以下研究领域生成创新的研究想法：

{query}"""
                else:
                    prompt = f"""Generate innovative research ideas for:

{query}"""

                print("[ideation] Computing embeddings for similarity analysis...")

                query_embedding = await get_embedding(query)

                if not query_embedding:
                    print("[ideation] Failed to get query embedding, generating ideas without similarity analysis")
                else:
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

        print(f"[paper_review] Received query: {query}")
        print(f"[paper_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        language = detect_language(query)
        print(f"[paper_review] Detected language: {language}")

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
