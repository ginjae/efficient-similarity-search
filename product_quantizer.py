import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class ProductQuantizer:
    def __init__(self, M=8, Ks=256, max_iter=100):
        self.M = M                  # 서브벡터 수
        self.Ks = Ks                # 각 서브공간의 codebook 크기 (보통 256)
        self.max_iter = max_iter
        self.codes = np.empty(0)
        self.codebooks = []         # 각 서브공간의 KMeans 모델

    def train(self, X):
        """ PQ 학습: 각 서브공간에서 KMeans 수행 """
        _, D = X.shape
        assert D % self.M == 0, "벡터 차원은 M으로 나누어져야 합니다."
        self.ds = D // self.M

        def train_kmeans(subvectors):
            kmeans = KMeans(n_clusters=self.Ks, max_iter=self.max_iter, n_init=1, random_state=42)
            kmeans.fit(subvectors)
            return kmeans

        subvectors_list = [X[:, m*self.ds:(m+1)*self.ds] for m in range(self.M)]
        self.codebooks = Parallel(n_jobs=-1)(delayed(train_kmeans)(sv) for sv in subvectors_list)

    def add(self, X):
        self.codes = self.encode(X)

    def encode(self, X):
        """ 각 벡터를 (M,) 길이의 code index로 변환 """
        N = X.shape[0]
        codes = np.empty((N, self.M), dtype=np.uint8)

        def encode_column(m):
            subvectors = X[:, m*self.ds:(m+1)*self.ds]
            return self.codebooks[m].predict(subvectors)

        results = Parallel(n_jobs=-1)(delayed(encode_column)(m) for m in range(self.M))

        for m in range(self.M):
            codes[:, m] = results[m]

        return codes

    def decode(self, codes):
        """ code index → 복원된 벡터 """
        N = codes.shape[0]
        X_recon = np.empty((N, self.M * self.ds), dtype=np.float32)

        for m in range(self.M):
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
        assert queries.shape[1] == self.M * self.ds
        B = queries.shape[0]
        N = self.codes.shape[0]

        def process_query(query):
            dist_tables = []
            for m in range(self.M):
                q_sub = query[m*self.ds:(m+1)*self.ds].reshape(1, -1)
                centroids = self.codebooks[m].cluster_centers_
                dists = np.linalg.norm(centroids - q_sub, axis=1) ** 2
                dist_tables.append(dists)

            distances = np.zeros(N, dtype=np.float32)
            for m in range(self.M):
                distances += dist_tables[m][self.codes[:, m]]

            topk_idx = np.argpartition(distances, topk)[:topk]
            topk_sorted = topk_idx[np.argsort(distances[topk_idx])]
            return distances[topk_sorted], topk_sorted

        results = Parallel(n_jobs=-1)(delayed(process_query)(queries[b]) for b in range(B))

        all_distances, all_indices = zip(*results)
        return list(all_distances), list(all_indices)

    def search_original(self, queries, X, topk=10):
        """
        query: (B, D) 형태의 원본 쿼리 벡터들
        X: (N, D) 압축되지 않은 데이터 벡터들
        topk: 상위 k개 인덱스 반환
        """
        assert queries.shape[1] == X.shape[1]
        B = queries.shape[0]
        N = X.shape[0]

        all_distances = []
        all_indices = []

        for b in range(B):
            query = queries[b]
            # 1. 거리 계산
            diff = X - query.reshape(1, -1)
            distances = np.sum(diff**2, axis=1)

            # 2. top-k 추출
            topk_idx = np.argpartition(distances, topk)[:topk]
            topk_sorted = topk_idx[np.argsort(distances[topk_idx])]
            all_distances.append(distances[topk_sorted])
            all_indices.append(topk_sorted)

        return all_distances, all_indices  # 각각 (B, topk) 형태의 리스트
