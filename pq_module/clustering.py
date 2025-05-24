from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
# from sklearn_extra.cluster import KMedoids  # 메모리 문제로 제외
RANDOM_SEED = 42

def train_clustering(method, subvectors, n_clusters, max_iter):
    if method == "k-means":
        return KMeans(
            n_clusters=n_clusters,
            init="random",              # k-means++ or random
            algorithm="elkan",          # elkan or lloyd
            max_iter=max_iter,          # maximum number of iterations
            n_init=1,                   # for fast training, just try once
            random_state=RANDOM_SEED    # random seed
        ).fit(subvectors)

    elif method == "k-means++":
        return KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            algorithm="elkan",
            max_iter=max_iter,
            n_init=1,
            random_state=RANDOM_SEED
        ).fit(subvectors)

    elif method == "mini-batch-k-means":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            batch_size=1024,
            max_iter=max_iter,
            n_init=1,
            random_state=RANDOM_SEED
        ).fit(subvectors)

    elif method == "bisecting-k-means":
        return BisectingKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            algorithm="elkan",
            bisecting_strategy="largest_cluster",   # biggest_inertia or largest_cluster
            max_iter=max_iter,
            n_init=1,
            random_state=RANDOM_SEED
        ).fit(subvectors)

# 메모리 사용량, 연산량 문제로 k-medoids는 사용 x
#     elif method == "k-medoids":
#         return KMedoids(
#             n_clusters=n_clusters,
#             method="alternate",         # alternate or pam
#             init="k-medoids++",         # random, heuristic, k-medoids++, or build
#             max_iter=max_iter,
#             random_state=RANDOM_SEED
#         ).fit(subvectors)

    else:
        raise ValueError(f"Unsupported clustering method: {method}")
