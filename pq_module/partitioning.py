import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def balanced_clusters_by_sum(vec, m):
    n = vec.size
    
    # 클러스터 개수 계산
    cluster_size = n // m

    # (인덱스, 값) 쌍을 값 내림차순으로 정렬
    idx_vals = sorted(enumerate(vec), key=lambda iv: iv[1], reverse=True)

    # 클러스터 합과 인덱스 초기화
    cluster_sums = np.zeros(m, dtype=float)
    clusters = [[] for _ in range(m)]

    # 그리디 배정: 가장 큰 값부터, 현재 합이 가장 작은 클러스터에 할당
    for idx, value in idx_vals:
        # 여유 슬롯이 있는 클러스터 중에서
        candidates = [i for i in range(m) if len(clusters[i]) < cluster_size]
        # 그중 합이 최소인 클러스터 선택
        j = min(candidates, key=lambda i: cluster_sums[i])
        clusters[j].append(idx)
        cluster_sums[j] += value

    return clusters


def train_partitioning(X, n_clusters):

#     pca = PCA(n_components=X.shape[1])
#     pca.fit_transform(X)
#     explained_variance_ratio = pca.explained_variance_
#     print(explained_variance_ratio)
#     print(np.var(X, axis=0))
    dim_variances = np.var(X, axis=0)
    subspace_dims = balanced_clusters_by_sum(dim_variances, n_clusters)
    
    return subspace_dims

# from sklearn.cluster import AgglomerativeClustering
# 
# def rebalance_clusters(clusters, m=16):
#     elements = [x for cluster in clusters for x in cluster]
#     n = len(elements)
#     k, r = divmod(n, m)
#     
#     result = []
#     start = 0
#     for i in range(m):
#         end = start + k + (1 if i < r else 0)
#         result.append(elements[start:end])
#         start = end
#     return result
# 
# def train_partitioning(X, n_clusters):
#     assert len(X.shape) == 2
# 
#     # 1. 차원 간 피어슨 상관계수 계산
#     corr_matrix = np.corrcoef(X.T)  # shape: (d, d)
#     corr_matrix = np.abs(corr_matrix)
#     
#     # 2. 거리 행렬로 변환
#     distance_matrix = 1 - corr_matrix
#     
#     # 3. Agglomerative Clustering 수행
#     clustering = AgglomerativeClustering(
#         n_clusters=n_clusters,
#         metric="precomputed",
#         linkage="average"
#     )
#     dim_labels = clustering.fit_predict(distance_matrix)
#     
#     # 4. 결과 정리: 클러스터별 차원 인덱스 그룹
#     clusters = [[] for _ in range(n_clusters)]
#     for dim_idx, label in enumerate(dim_labels):
#         clusters[label].append(dim_idx)
# 
#     print(clusters)
#     clusters = rebalance_clusters(clusters)
#     print(clusters)
# 
#     return clusters
