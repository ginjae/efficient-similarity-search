import numpy as np
from joblib import Parallel, delayed
from .clustering import train_clustering

class OptimizedProductQuantizer:
    def __init__(self, clustering="k-means", m=16, n_clusters=256, max_iter=100, opq_iter=8):
        self.clustering = clustering
        self.m = m                      # number of sub-vectors
        self.n_clusters = n_clusters    # codebook size per subspace
        self.max_iter = max_iter        # max k-means iterations
        self.opq_iter = opq_iter        # number of OPQ alternating iterations
        self.codes = None               # quantized codes of training data
        self.codebooks = []             # list of trained KMeans models
        self.R = None                   # learned rotation matrix

    def train(self, X):
        """ PQ 학습: 회전 행렬과 Kmeans를 교대로 최적화 """
        _, D = X.shape
        assert D % self.m == 0, "벡터 차원은 M으로 나누어져야 합니다."
        self.ds = D // self.m

        # initialize rotation as identity
        self.R = np.eye(D, dtype=X.dtype)

        for it in range(self.opq_iter):
            print(f"it = {it}")
            # 1) Rotate dataset
            X_rot = X.dot(self.R)

            # 2) Train PQ codebooks on rotated data
            subvectors = [X_rot[:, i*self.ds:(i+1)*self.ds] for i in range(self.m)]
            self.codebooks = Parallel(n_jobs=-1)(
                    delayed(train_clustering)(self.clustering, sv, self.n_clusters, self.max_iter)
                    for sv in subvectors
            )

            # 3) Encode & reconstruct rotated data
            codes = self.encode(X)
            X_rot_hat = self.decode_rotated(codes)

            # 4) Update rotation via orthogonal Procrustes
            M = X.T.dot(X_rot_hat)
            U, _, Vt = np.linalg.svd(M)
            self.R = U.dot(Vt)

        # store final codes
        self.codes = self.encode(X)

    def add(self, X):
        self.codes = self.encode(X)

    def encode(self, X):
        """ 각 벡터에 회전 행렬을 곱한 후, (M,) 길이의 code index로 변환 """
        X_rot = X.dot(self.R)
        N = X.shape[0]
        codes = np.empty((N, self.m), dtype=np.uint8)

        def encode_sub(i):
            sub = X_rot[:, i*self.ds:(i+1)*self.ds]
            return self.codebooks[i].predict(sub)

        results = Parallel(n_jobs=-1)(delayed(encode_sub)(i) for i in range(self.m))
        for i, r in enumerate(results):
            codes[:, i] = r
        return codes

    def decode_rotated(self, codes):
        """ PQ codes를 회전된 벡터로 decode """
        N = codes.shape[0]
        D = self.m * self.ds
        X_hat = np.empty((N, D), dtype=np.float32)
        for i in range(self.m):
            centroids = self.codebooks[i].cluster_centers_
            X_hat[:, i*self.ds:(i+1)*self.ds] = centroids[codes[:, i]]
        return X_hat

    def decode(self, codes):
        """ PQ codes를 원래 벡터로 decode """
        X_rot_hat = self.decode_rotated(codes)
        # inverse rotate
        return X_rot_hat.dot(self.R.T)

    def search(self, queries, topk=10):
        """
        Approximate nearest neighbor search using Asymmetric Distance Computation (ADC).
        queries: (B, D) original query vectors
        Returns distances and indices of topk hits for each query.
        """
        assert queries.shape[1] == self.R.shape[0], "Query dimensionality must match"
        # rotate queries
        Q_rot = queries.dot(self.R)
        return self._search_rotated(Q_rot, topk)

    def _search_rotated(self, Q_rot, topk):
        B = Q_rot.shape[0]
        N = self.codes.shape[0]
        ds = self.ds

        def process_query(q):
            # build distance tables for each subspace
            tables = []
            for i in range(self.m):
                sub_q = q[i*ds:(i+1)*ds].reshape(1, -1)
                centroids = self.codebooks[i].cluster_centers_
                tables.append(np.linalg.norm(centroids - sub_q, axis=1)**2)

            # accumulate distances
            dists = np.zeros(N, dtype=np.float32)
            for i in range(self.m):
                dists += tables[i][self.codes[:, i]]

            # select topk
            idx = np.argpartition(dists, topk)[:topk]
            idx = idx[np.argsort(dists[idx])]
            return dists[idx], idx

        results = Parallel(n_jobs=-1)(delayed(process_query)(Q_rot[b]) for b in range(B))
        dists, idxs = zip(*results)
        return list(dists), list(idxs)

