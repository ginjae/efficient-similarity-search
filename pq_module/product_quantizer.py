import numpy as np
from joblib import Parallel, delayed
from .clustering import train_clustering

class ProductQuantizer:
    def __init__(self, clustering="k-means", m=16, n_clusters=256, max_iter=100):
        self.clustering = clustering
        self.m = m                      # 서브벡터 수
        self.n_clusters = n_clusters    # 각 서브공간의 codebook 크기 (cluster 수)
        self.max_iter = max_iter
        self.codes = None               # encoded (quantized) X_base 보관
        self.codebooks = []             # 각 서브공간의 KMeans 모델

    def train(self, X):
        """ PQ 학습: 각 서브공간에서 KMeans 수행 """
        _, D = X.shape
        assert D % self.m == 0, "벡터 차원은 M으로 나누어져야 합니다."
        self.ds = D // self.m

        # Fixed and Disjointed Partitioning
        subvectors_list = [X[:, m*self.ds:(m+1)*self.ds] for m in range(self.m)]

        self.codebooks = Parallel(n_jobs=-1)(
                delayed(train_clustering)(self.clustering, sv, self.n_clusters, self.max_iter)
                for sv in subvectors_list
        )

    def add(self, X):
        self.codes = self.encode(X)

    def encode(self, X):
        """ 각 벡터를 (M,) 길이의 code index로 변환 """
        N = X.shape[0]
        codes = np.empty((N, self.m), dtype=np.uint8)

        def encode_column(m):
            subvectors = X[:, m*self.ds:(m+1)*self.ds]
            return self.codebooks[m].predict(subvectors)

        results = Parallel(n_jobs=-1)(delayed(encode_column)(m) for m in range(self.m))

        for m in range(self.m):
            codes[:, m] = results[m]

        return codes

    def decode(self, codes):
        """ code index → 복원된 벡터 """
        N = codes.shape[0]
        X_recon = np.empty((N, self.m * self.ds), dtype=np.float32)

        for m in range(self.m):
            centroids = self.codebooks[m].cluster_centers_
            X_recon[:, m*self.ds:(m+1)*self.ds] = centroids[codes[:, m]]

        return X_recon

    def search(self, queries, topk=10):
        """
        사전에 fit 필요
        query: (B, D) 형태의 원본 쿼리 벡터들
        codes: (N, M) PQ 인코딩된 데이터셋
        topk: 상위 k개 인덱스 반환
        """
        assert queries.shape[1] == self.m * self.ds
        B = queries.shape[0]
        N = self.codes.shape[0]

        def process_query(query):
            dist_tables = []
            for m in range(self.m):
                q_sub = query[m*self.ds:(m+1)*self.ds].reshape(1, -1)
                centroids = self.codebooks[m].cluster_centers_
                dists = np.linalg.norm(centroids - q_sub, axis=1) ** 2
                dist_tables.append(dists)

            distances = np.zeros(N, dtype=np.float32)
            for m in range(self.m):
                distances += dist_tables[m][self.codes[:, m]]

            topk_idx = np.argpartition(distances, topk)[:topk]
            topk_sorted = topk_idx[np.argsort(distances[topk_idx])]
            return distances[topk_sorted], topk_sorted

        results = Parallel(n_jobs=-1)(delayed(process_query)(queries[b]) for b in range(B))

        all_distances, all_indices = zip(*results)
        return list(all_distances), list(all_indices)
