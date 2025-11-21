"""
基于 Embedding 的文献筛选和聚类工具
使用 text-embedding-v4 计算相关性并进行聚类
"""
import os
import asyncio
import numpy as np
from typing import List, Optional, Tuple, Dict
from openai import AsyncOpenAI
from base import PaperResult
from multi_search import multi_source_search
from dotenv import load_dotenv


load_dotenv()

# 初始化 embedding 客户端
embedding_base_url = os.getenv("SCI_EMBEDDING_BASE_URL")
embedding_api_key = os.getenv("SCI_EMBEDDING_API_KEY")
sci_model_base_url = os.getenv("SCI_MODEL_BASE_URL")
sci_model_api_key = os.getenv("SCI_MODEL_API_KEY")

if embedding_base_url and embedding_api_key:
    embedding_client = AsyncOpenAI(
        base_url=embedding_base_url,
        api_key=embedding_api_key
    )
elif sci_model_base_url and sci_model_api_key:
    # 如果没有单独的 embedding 客户端，使用主模型客户端
    embedding_client = AsyncOpenAI(
        base_url=sci_model_base_url,
        api_key=sci_model_api_key
    )
else:
    raise ValueError("需要设置 SCI_EMBEDDING_BASE_URL 和 SCI_EMBEDDING_API_KEY 或 SCI_MODEL_BASE_URL 和 SCI_MODEL_API_KEY")


async def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
        
    Returns:
        嵌入向量
    """
    if not text:
        return []
    
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


def _build_paper_text(paper: PaperResult) -> str:
    """
    构建论文的文本表示（用于 embedding）
    
    Args:
        paper: 论文结果对象
        
    Returns:
        论文的文本表示
    """
    parts = []
    
    if paper.title:
        parts.append(paper.title)
    
    if paper.abstract:
        parts.append(paper.abstract)
    
    if paper.authors:
        parts.append(", ".join(paper.authors[:3]))  # 只包含前3个作者
    
    return " ".join(parts)


async def filter_by_relevance(
    papers: List[PaperResult],
    query: str,
    similarity_threshold: float = 0.3,
    top_k: Optional[int] = None,
    return_embeddings: bool = False
) -> List[Tuple[PaperResult, float, Optional[List[float]]]]:
    """
    根据与查询主题的相关性筛选文献，使用 text-embedding-v4
    
    Args:
        papers: 论文列表
        query: 查询主题
        similarity_threshold: 相似度阈值，低于此值的论文将被过滤
        top_k: 如果指定，只返回相似度最高的 k 篇论文
        return_embeddings: 是否返回 embedding 向量（用于后续聚类）
        
    Returns:
        筛选后的论文列表，每个元素是 (论文, 相似度分数, embedding向量) 的元组，按相似度降序排列
        如果 return_embeddings=False，embedding 向量为 None
    """
    if not papers:
        return []
    
    # 获取查询的 embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        print("无法获取查询的 embedding，返回所有论文")
        if return_embeddings:
            return [(paper, 0.0, None) for paper in papers]
        return [(paper, 0.0, None) for paper in papers]
    
    # 批量计算所有论文的 embedding 和相似度
    paper_similarities = []
    
    # 构建论文文本
    paper_texts = [_build_paper_text(paper) for paper in papers]
    
    # 并行获取所有论文的 embedding
    tasks = [get_embedding(text) for text in paper_texts]
    embeddings = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 计算相似度
    for paper, embedding in zip(papers, embeddings):
        if isinstance(embedding, Exception) or not embedding:
            similarity = 0.0
            paper_embedding = None
        else:
            similarity = cosine_similarity(query_embedding, embedding)
            paper_embedding = embedding if return_embeddings else None
        
        if similarity >= similarity_threshold:
            paper_similarities.append((paper, similarity, paper_embedding))
    
    # 按相似度降序排序
    paper_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 如果指定了 top_k，只返回前 k 个
    if top_k is not None and top_k > 0:
        paper_similarities = paper_similarities[:top_k]
    
    return paper_similarities


def cluster_papers(
    papers_with_similarity: List[Tuple[PaperResult, float, Optional[List[float]]]],
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2,
    use_embeddings: bool = True
) -> Dict[int, List[Tuple[PaperResult, float, Optional[List[float]]]]]:
    """
    对论文进行聚类，直接使用 text-embedding-v4 的 embedding 向量

    Args:
        papers_with_similarity: 论文、相似度分数和 embedding 向量的列表
        n_clusters: 聚类数量，如果为 None 则自动确定
        min_cluster_size: 最小聚类大小
        use_embeddings: 是否使用 embedding 向量进行聚类（True）或使用简单方法（False）

    Returns:
        聚类结果字典，key 是聚类编号，value 是该聚类的论文列表
    """
    if len(papers_with_similarity) < min_cluster_size:
        # 如果论文数量太少，返回单个聚类
        return {0: papers_with_similarity}

    # 检查是否有 embedding 向量
    embeddings_available = use_embeddings and all(
        item[2] is not None and len(item[2]) > 0
        for item in papers_with_similarity
    )

    if embeddings_available:
        # 使用 embedding 向量进行聚类
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # 提取 embedding 向量
            embeddings = [item[2] for item in papers_with_similarity]
            embeddings_array = np.array(embeddings)

            # 检查 embedding 向量是否有效
            if embeddings_array.size == 0 or np.isnan(embeddings_array).any() or np.isinf(embeddings_array).any():
                raise ValueError("Embedding 向量包含无效值（NaN 或 Inf）")

            # 标准化 embedding（KMeans 对尺度敏感）
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_array)

            # 确定聚类数量
            if n_clusters is None:
                # 使用简单的启发式方法：聚类数量 = sqrt(论文数量/2)，但至少2个，最多10个
                n_clusters = max(2, min(10, int(np.sqrt(len(papers_with_similarity) / 2))))

            n_clusters = min(n_clusters, len(papers_with_similarity))  # 不能超过论文数量

            # 如果聚类数量为1，直接返回单个聚类
            if n_clusters == 1:
                return {0: papers_with_similarity}

            # 执行 KMeans 聚类（基于 embedding 向量）
            # 使用更保守的参数设置，避免某些环境下的问题
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
                cluster_labels = kmeans.fit_predict(embeddings_scaled)
            except Exception as kmeans_error:
                # 如果标准 KMeans 失败，直接切换到简单聚类方法
                # 不再尝试简化参数，因为问题可能出在 sklearn 内部的 threadpoolctl
                error_msg = str(kmeans_error)
                if 'NoneType' in error_msg or 'split' in error_msg or 'threadpoolctl' in error_msg:
                    print(f"提示: sklearn 聚类遇到兼容性问题（{type(kmeans_error).__name__}），自动切换到基于 embedding 的简单聚类方法")
                else:
                    print(f"提示: sklearn 聚类失败 ({type(kmeans_error).__name__}: {error_msg})，使用基于 embedding 的简单聚类方法")
                return _simple_cluster_by_embedding(papers_with_similarity, n_clusters, min_cluster_size)

            # 组织聚类结果
            clusters = {}
            for idx, (paper_data, label) in enumerate(zip(papers_with_similarity, cluster_labels)):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(paper_data)

            # 过滤掉太小的聚类
            filtered_clusters = {}
            cluster_id = 0
            for label, cluster_papers in clusters.items():
                if len(cluster_papers) >= min_cluster_size:
                    filtered_clusters[cluster_id] = cluster_papers
                    cluster_id += 1
                else:
                    # 将小聚类合并到最近的聚类中
                    if filtered_clusters:
                        # 找到最大的聚类并合并
                        largest_cluster = max(filtered_clusters.values(), key=len)
                        largest_cluster.extend(cluster_papers)
                    else:
                        filtered_clusters[cluster_id] = cluster_papers
                        cluster_id += 1

            return filtered_clusters

        except ImportError as e:
            # sklearn 未安装
            print(f"提示: sklearn 未安装，使用基于 embedding 的简单聚类方法")
            return _simple_cluster_by_embedding(papers_with_similarity, n_clusters, min_cluster_size)
        except (AttributeError, ValueError, RuntimeError) as e:
            # sklearn 内部错误或数据问题，使用回退方法
            error_msg = str(e)
            if 'NoneType' in error_msg or 'split' in error_msg:
                print(f"提示: sklearn 聚类遇到兼容性问题，自动切换到基于 embedding 的简单聚类方法")
            else:
                print(f"提示: sklearn 聚类失败 ({type(e).__name__}: {error_msg})，使用基于 embedding 的简单聚类方法")
            return _simple_cluster_by_embedding(papers_with_similarity, n_clusters, min_cluster_size)
        except Exception as e:
            # 其他未知错误
            print(f"提示: 聚类过程遇到问题 ({type(e).__name__}: {str(e)})，使用基于 embedding 的简单聚类方法")
            return _simple_cluster_by_embedding(papers_with_similarity, n_clusters, min_cluster_size)
    else:
        # 如果没有 embedding 向量，使用简单的基于相似度的聚类方法
        print("未提供 embedding 向量，使用基于相似度的简单聚类方法")
        return _simple_cluster_by_similarity(papers_with_similarity, n_clusters, min_cluster_size)


def _simple_cluster_by_embedding(
    papers_with_similarity: List[Tuple[PaperResult, float, Optional[List[float]]]],
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2
) -> Dict[int, List[Tuple[PaperResult, float, Optional[List[float]]]]]:
    """
    基于 embedding 向量的简单 K-means 聚类（不依赖 sklearn）
    
    使用 embedding 向量之间的余弦距离进行聚类
    """
    if len(papers_with_similarity) < min_cluster_size:
        return {0: papers_with_similarity}
    
    # 检查是否有 embedding 向量
    papers_with_embeddings = [
        item for item in papers_with_similarity 
        if item[2] is not None and len(item[2]) > 0
    ]
    
    if len(papers_with_embeddings) < min_cluster_size:
        # 如果没有足够的 embedding，回退到基于相似度的方法
        return _simple_cluster_by_similarity(papers_with_similarity, n_clusters, min_cluster_size)
    
    # 提取 embedding 向量
    embeddings = np.array([item[2] for item in papers_with_embeddings])
    
    # 确定聚类数量
    if n_clusters is None:
        n_clusters = max(2, min(5, int(np.sqrt(len(papers_with_embeddings) / 2))))
    
    n_clusters = min(n_clusters, len(papers_with_embeddings))
    
    # 简单的 K-means 实现（基于余弦距离）
    # 初始化：随机选择 n_clusters 个点作为初始中心
    np.random.seed(42)
    indices = np.random.choice(len(papers_with_embeddings), n_clusters, replace=False)
    centroids = embeddings[indices]
    
    # 迭代聚类（最多 20 次）
    max_iterations = 20
    for iteration in range(max_iterations):
        # 计算每个点到最近中心的距离（使用余弦距离）
        clusters = [[] for _ in range(n_clusters)]
        
        for idx, embedding in enumerate(embeddings):
            # 计算与所有中心的余弦相似度
            similarities = [
                cosine_similarity(embedding.tolist(), centroid.tolist())
                for centroid in centroids
            ]
            # 选择相似度最高的中心（余弦距离 = 1 - 余弦相似度）
            closest_center = np.argmax(similarities)
            clusters[closest_center].append(idx)
        
        # 更新中心点（取每个聚类的平均 embedding）
        new_centroids = []
        for cluster_indices in clusters:
            if len(cluster_indices) > 0:
                cluster_embeddings = embeddings[cluster_indices]
                new_centroid = np.mean(cluster_embeddings, axis=0)
                new_centroids.append(new_centroid)
            else:
                # 如果聚类为空，保持原中心
                new_centroids.append(centroids[len(new_centroids)])
        
        new_centroids = np.array(new_centroids)
        
        # 检查是否收敛（中心点变化很小）
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    # 组织聚类结果
    result_clusters = {}
    for cluster_id, cluster_indices in enumerate(clusters):
        if len(cluster_indices) >= min_cluster_size:
            result_clusters[cluster_id] = [papers_with_embeddings[idx] for idx in cluster_indices]
    
    # 处理未分配的论文（如果有）
    assigned_indices = set()
    for cluster_indices in clusters:
        assigned_indices.update(cluster_indices)
    
    unassigned = [
        papers_with_embeddings[idx] 
        for idx in range(len(papers_with_embeddings))
        if idx not in assigned_indices
    ]
    
    # 将未分配的论文添加到最近的聚类
    if unassigned and result_clusters:
        for paper_data in unassigned:
            paper, similarity, embedding = paper_data
            # 找到最近的聚类
            similarities_to_clusters = []
            for cluster_id, cluster_papers in result_clusters.items():
                # 计算与聚类中心的平均相似度
                cluster_embeddings = [item[2] for item in cluster_papers if item[2] is not None]
                if cluster_embeddings:
                    avg_embedding = np.mean(cluster_embeddings, axis=0)
                    sim = cosine_similarity(embedding, avg_embedding.tolist())
                    similarities_to_clusters.append((cluster_id, sim))
            
            if similarities_to_clusters:
                closest_cluster_id = max(similarities_to_clusters, key=lambda x: x[1])[0]
                result_clusters[closest_cluster_id].append(paper_data)
            else:
                # 如果没有聚类，创建新聚类
                result_clusters[len(result_clusters)] = [paper_data]
    
    # 处理没有 embedding 的论文
    papers_without_embeddings = [
        item for item in papers_with_similarity 
        if item[2] is None or len(item[2]) == 0
    ]
    
    if papers_without_embeddings:
        # 将它们添加到最大的聚类中
        if result_clusters:
            largest_cluster = max(result_clusters.values(), key=len)
            largest_cluster.extend(papers_without_embeddings)
        else:
            result_clusters[0] = papers_without_embeddings
    
    # 重新编号聚类
    filtered_clusters = {}
    cluster_id = 0
    for cluster_papers in result_clusters.values():
        if len(cluster_papers) >= min_cluster_size:
            filtered_clusters[cluster_id] = cluster_papers
            cluster_id += 1
        else:
            # 合并到最近的聚类
            if filtered_clusters:
                largest_cluster = max(filtered_clusters.values(), key=len)
                largest_cluster.extend(cluster_papers)
            else:
                filtered_clusters[cluster_id] = cluster_papers
                cluster_id += 1
    
    return filtered_clusters if filtered_clusters else {0: papers_with_similarity}


def _simple_cluster_by_similarity(
    papers_with_similarity: List[Tuple[PaperResult, float, Optional[List[float]]]],
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2
) -> Dict[int, List[Tuple[PaperResult, float, Optional[List[float]]]]]:
    """
    简单的基于相似度分数的聚类方法（当没有 embedding 时使用）
    
    使用相似度分数进行简单的区间聚类
    """
    if len(papers_with_similarity) < min_cluster_size:
        return {0: papers_with_similarity}
    
    # 按相似度排序
    sorted_papers = sorted(papers_with_similarity, key=lambda x: x[1], reverse=True)
    
    # 确定聚类数量
    if n_clusters is None:
        n_clusters = max(2, min(5, int(np.sqrt(len(sorted_papers) / 2))))
    
    n_clusters = min(n_clusters, len(sorted_papers))
    
    # 简单的基于相似度区间的聚类
    # 将相似度范围分成 n_clusters 个区间
    similarities = [item[1] for item in sorted_papers]
    min_sim = min(similarities)
    max_sim = max(similarities)
    
    if max_sim == min_sim:
        # 如果所有相似度相同，返回单个聚类
        return {0: sorted_papers}
    
    # 创建聚类
    clusters = {}
    interval_size = (max_sim - min_sim) / n_clusters
    
    for paper_data in sorted_papers:
        paper, similarity, embedding = paper_data
        # 确定应该属于哪个聚类
        cluster_idx = min(int((similarity - min_sim) / interval_size), n_clusters - 1)
        
        if cluster_idx not in clusters:
            clusters[cluster_idx] = []
        clusters[cluster_idx].append(paper_data)
    
    # 过滤和重组聚类
    filtered_clusters = {}
    cluster_id = 0
    for cluster_idx, cluster_papers in sorted(clusters.items()):
        if len(cluster_papers) >= min_cluster_size:
            filtered_clusters[cluster_id] = cluster_papers
            cluster_id += 1
        else:
            # 合并到最近的聚类
            if filtered_clusters:
                largest_cluster = max(filtered_clusters.values(), key=len)
                largest_cluster.extend(cluster_papers)
            else:
                filtered_clusters[cluster_id] = cluster_papers
                cluster_id += 1
    
    return filtered_clusters


async def search_filter_and_cluster(
    query: str,
    max_results_per_source: int = 10,
    similarity_threshold: float = 0.3,
    top_k: Optional[int] = None,
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2
) -> Dict[str, any]:
    """
    搜索文献，按相关性筛选，并进行聚类
    
    Args:
        query: 搜索主题
        max_results_per_source: 每个数据源的最大返回结果数
        similarity_threshold: 相似度阈值，低于此值的论文将被过滤
        top_k: 如果指定，只返回相似度最高的 k 篇论文
        n_clusters: 聚类数量，如果为 None 则自动确定
        min_cluster_size: 最小聚类大小
        
    Returns:
        包含以下字段的字典：
        - 'query': 查询主题
        - 'total_papers': 搜索到的总论文数
        - 'filtered_papers': 筛选后的论文列表，每个元素是包含完整信息的字典：
            {
                'title': 论文标题,
                'authors': 作者列表,
                'abstract': 摘要,
                'year': 发表年份,
                'doi': DOI,
                'url': 链接,
                'pdf_url': PDF链接,
                'source': 数据源,
                'similarity': 相似度分数,
                'arxiv_id': arXiv ID (如果有),
                'category': 分类 (如果有),
                'venue': 会议/期刊 (如果有),
                'published_date': 发布日期 (如果有)
            }
        - 'clusters': 聚类结果字典，key 是聚类编号（字符串），value 是论文列表（格式同 filtered_papers）
        - 'cluster_count': 聚类数量
    """
    # 1. 多源搜索
    print(f"正在搜索主题: {query}...")
    all_papers = await multi_source_search(query, max_results_per_source=max_results_per_source)
    print(f"搜索到 {len(all_papers)} 篇论文")
    
    # 2. 按相关性筛选（同时获取 embedding 向量用于聚类）
    print(f"正在按相关性筛选（阈值: {similarity_threshold}）...")
    filtered_papers = await filter_by_relevance(
        all_papers,
        query,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        return_embeddings=True  # 返回 embedding 向量用于聚类
    )
    print(f"筛选后剩余 {len(filtered_papers)} 篇论文")
    
    # 3. 使用 embedding 向量进行聚类
    print("正在使用 embedding 向量进行聚类...")
    clusters = cluster_papers(
        filtered_papers,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        use_embeddings=True  # 使用 embedding 向量进行聚类
    )
    print(f"聚类完成，共 {len(clusters)} 个聚类")
    
    # 将论文信息转换为字典格式，包含完整的文献信息
    def format_paper_info(paper_data):
        """将论文数据格式化为字典"""
        paper, similarity, embedding = paper_data
        return {
            'title': paper.title or '',
            'authors': paper.authors or [],
            'abstract': paper.abstract or '',
            'year': paper.year,
            'doi': paper.doi or '',
            'url': paper.url or '',
            'pdf_url': paper.pdf_url or '',
            'source': paper.source or '',
            'similarity': similarity,
            'arxiv_id': paper.arxiv_id or '',
            'category': paper.category or '',
            'venue': paper.venue or '',
            'published_date': paper.published_date or ''
        }
    
    # 格式化筛选后的论文列表
    formatted_filtered_papers = [format_paper_info(item) for item in filtered_papers]
    
    # 格式化聚类结果
    formatted_clusters = {}
    for cluster_id, papers_in_cluster in clusters.items():
        formatted_clusters[str(cluster_id)] = [format_paper_info(item) for item in papers_in_cluster]
    
    return {
        'query': query,
        'total_papers': len(all_papers),
        'filtered_papers': formatted_filtered_papers,
        'clusters': formatted_clusters,
        'cluster_count': len(clusters)
    }


# 使用示例
if __name__ == "__main__":
    async def main():
        query = "叶面积指数"
        result = await search_filter_and_cluster(
            query=query,
            max_results_per_source=17,
            similarity_threshold=0.5,
            top_k=20,  # 只保留相似度最高的20篇
            n_clusters=None,  # 自动确定聚类数量
            min_cluster_size=2
        )
        
        print(f"\n查询主题: {result['query']}")
        print(f"搜索到论文总数: {result['total_papers']}")
        print(f"筛选后论文数: {len(result['filtered_papers'])}")
        print(f"聚类数量: {result['cluster_count']}\n")
        
        # 显示每个聚类的论文
        for cluster_id, papers in result['clusters'].items():
            print(f"聚类 {cluster_id} ({len(papers)} 篇论文):")
            for paper_info in papers:
                print(f"  - {paper_info['title'][:60]}... (相似度: {paper_info['similarity']:.3f})")
                print(f"    作者: {', '.join(paper_info['authors'][:3]) if paper_info['authors'] else '未知'}")
                print(f"    年份: {paper_info['year']}, DOI: {paper_info['doi']}")
            print()
    
    asyncio.run(main())

