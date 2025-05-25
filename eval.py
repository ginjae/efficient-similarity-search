"""

[EVALUATION]

ANN 벤치마크 데이터셋 대상으로 
1. accuracy (recall@k)
2. memory usage
3. search time
4. training time


"""

import sys
import os
import subprocess
import psutil
import threading
import time
import csv
import numpy as np

# PQ
from pq_module.product_quantizer import ProductQuantizer
from pq_module.adaptive_product_quantizer import AdaptiveProductQuantizer
from pq_module.optimized_product_quantizer import OptimizedProductQuantizer

# faiss
import faiss
from faiss.contrib.vecs_io import ivecs_read, fvecs_read, ivecs_write

# fashion-mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


""" Preparation """
def download_dataset(name):
    dest_dir = "./datasets"
    os.makedirs(dest_dir, exist_ok=True)
    target_dir = os.path.join(dest_dir, name)
    
    if os.path.exists(target_dir):
        print(f"[INFO] '{target_dir}' 디렉토리가 이미 존재합니다. 다운로드를 생략합니다.")
        return target_dir

    url = f"ftp://ftp.irisa.fr/local/texmex/corpus/{name}.tar.gz"
    archive_path = os.path.join(dest_dir, f"{name}.tar.gz")

    print(f"[INFO] {name} 데이터셋 다운로드 중... ({url})")
    subprocess.run(f"wget -O {archive_path} {url}", shell=True, check=True)

    print(f"[INFO] 압축 해제 중... ({archive_path})")
    subprocess.run(f"tar -xzf {archive_path} -C {dest_dir}", shell=True, check=True)

    print(f"[INFO] 다운로드 및 압축 해제 완료: {target_dir}")
    subprocess.run(f"rm {archive_path}", shell=True, check=True)

    return target_dir

def load_dataset(name):
    if name == "fashion-mnist":
        fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        scaler = StandardScaler()
        X = scaler.fit_transform(fashion["data"].astype(np.float32))
        xt, xq = train_test_split(X, test_size=10_000, random_state=20211061)
        xt, xb = train_test_split(xt, test_size=50_000, random_state=20211061)
        xt, _ = train_test_split(xt, test_size=5_000, random_state=20211061)
        _, gt = exact_search(xq, xb, 100)
    elif name == "glove":
        X = []
        with open("./datasets/glove/glove.6B.300d.txt", "r") as f:
            for i, line in enumerate(f):
                if i >= 111_000:
                    break
                parts = line.strip().split()
                vector = np.array(parts[1:], dtype=np.float32)
                X.append(vector)
        X = np.array(X, dtype=np.float32)
        xt, xq = train_test_split(X, test_size=1_000, random_state=20211061)
        xt, xb = train_test_split(xt, test_size=100_000, random_state=20211061)
        _, gt = exact_search(xq, xb, 100)
    else:
        xt = fvecs_read(f"datasets/{name}/{name}_learn.fvecs")
        xb = fvecs_read(f"datasets/{name}/{name}_base.fvecs")
        xq = fvecs_read(f"datasets/{name}/{name}_query.fvecs")
        gt = ivecs_read(f"datasets/{name}/{name}_groundtruth.ivecs")

    return { "train": xt, "base": xb, "query": xq, "gt": gt }

def save_groundtruth(base, query, filename):
    _, topk_idx = exact_search(query, base, 100)
    ivecs_write(filename, np.array(topk_idx, dtype=np.int32))


""" Calculation """
def get_hit(a: set, b: set):
    return len(a & b)

def get_recall(a, b, k):
    cumulative_recall = 0
    n = len(a)
    for i in range(n):
        cumulative_recall += get_hit(set(a[i]), set(b[i])) / k
    return cumulative_recall / n

def exact_search(queries, X, topk=10):
    """
    query: (B, D) 형태의 원본 쿼리 벡터들
    X: (N, D) 압축되지 않은 데이터 벡터들
    topk: 상위 k개 인덱스 반환
    """
    assert queries.shape[1] == X.shape[1]
    B = queries.shape[0]

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

""" Evaluation """
def exact_result(base, query, gt):
    print("🔍 exact search start")
    search_start = time.perf_counter()
    _, topk_idx = exact_search(query, base, 100)
    search_end = time.perf_counter()
    print(f"⌛ exact search time: {search_end - search_start}")
    print(f"🎯 [EXACT] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def pq_result(train, base, query, gt, clustering, m):
    pq = ProductQuantizer(clustering=clustering, m=m)
    print(f"🧠 {clustering} pq training start")
    training_start = time.perf_counter()
    pq.train(train)
    pq.add(base)
    training_end = time.perf_counter()
    print(f"⏱️ {clustering} pq training time: {training_end - training_start}")

    print(f"🔍 {clustering} pq search start")
    search_start = time.perf_counter()
    _, topk_idx = pq.search(query, topk=100)
    search_end = time.perf_counter()
    print(f"⌛ {clustering} pq search time: {search_end - search_start}")
    print(f"🎯 [{clustering.upper()} PQ] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def apq_result(train, base, query, gt, clustering, m):
    pq = AdaptiveProductQuantizer(clustering=clustering, m=m)
    print(f"🧠 {clustering} apq training start")
    training_start = time.perf_counter()
    pq.train(train)
    pq.add(base)
    training_end = time.perf_counter()
    print(f"⏱️ {clustering} apq training time: {training_end - training_start}")

    print(f"🔍 {clustering} apq search start")
    search_start = time.perf_counter()
    _, topk_idx = pq.search(query, topk=100)
    search_end = time.perf_counter()
    print(f"⌛ {clustering} apq search time: {search_end - search_start}")
    print(f"🎯 [{clustering.upper()} APQ] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def opq_result(train, base, query, gt, clustering, m):
    pq = OptimizedProductQuantizer(clustering=clustering, m=m)
    print(f"🧠 {clustering} opq training start")
    training_start = time.perf_counter()
    pq.train(train)
    pq.add(base)
    training_end = time.perf_counter()
    print(f"⏱️ {clustering} opq training time: {training_end - training_start}")

    print(f"🔍 {clustering} opq search start")
    search_start = time.perf_counter()
    _, topk_idx = pq.search(query, topk=100)
    search_end = time.perf_counter()
    print(f"⌛ {clustering} opq search time: {search_end - search_start}")
    print(f"🎯 [{clustering.upper()} OPQ] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def faiss_pq_result(train, base, query, gt, m):
    indexPQ = faiss.IndexPQ(train.shape[1], m, 8, faiss.METRIC_L2)
    print("🧠 faiss pq training start")
    training_start = time.perf_counter()
    indexPQ.train(train)
    indexPQ.add(base)
    training_end = time.perf_counter()
    print(f"⏱️ faiss pq training time: {training_end - training_start}")

    print("🔍 faiss pq search start")
    search_start = time.perf_counter()
    _, topk_idx = indexPQ.search(query, 100)
    search_end = time.perf_counter()
    print(f"⌛ faiss pq search time: {search_end - search_start}")
    print(f"🎯 [FAISS PQ] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def faiss_opq_result(train, base, query, gt, m):
    opq_matrix = faiss.OPQMatrix(train.shape[1], m)
    indexPQ = faiss.IndexPQ(train.shape[1], m, 8, faiss.METRIC_L2)
    index = faiss.IndexPreTransform(opq_matrix, indexPQ)
    print("🧠 faiss opq training start")
    training_start = time.perf_counter()
    index.train(train)
    index.add(base)
    training_end = time.perf_counter()
    print(f"⏱️ faiss opq training time: {training_end - training_start}")

    print("🔍 faiss opq search start")
    search_start = time.perf_counter()
    _, topk_idx = index.search(query, 100)
    search_end = time.perf_counter()
    print(f"⌛ faiss opq search time: {search_end - search_start}")
    print(f"🎯 [FAISS OPQ] RECALL: {get_recall(topk_idx, gt, 100)}")
    print()

def evaluate(func,filename, *args):
    thread = threading.Thread(target=func, args=args)
    thread.start()

    p = psutil.Process()
    records = []
    timestamp = 0
    while thread.is_alive():
        memory_usage = p.memory_info().rss
        records.append([timestamp, memory_usage])
        timestamp += 1
        time.sleep(1)

    thread.join()

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "memory_usage_bytes"])
        writer.writerows(records)


""" main """
if __name__ == "__main__":
#     download_dataset("sift")
#     download_dataset("gist")
    
#     dataset_name = "sift"
#     dataset_name = "gist"
#     dataset_name = "deep"

    if len(sys.argv) != 2:
        print("Usage: python eval.py <dataset_name>")
        sys.exit(1)
    
    arg = sys.argv[1].lower()
    if arg != "sift" and arg != "gist" and arg != "deep" and arg != "fashion-mnist" and arg != "glove" and arg != "gist_m":
        print('Available Datasets: "sift", "gist", "deep", "fashion-mnist", "glove"')
        sys.exit(1)


    if arg == "gist_m":
        dataset_name = "gist"
    else:
        dataset_name = arg
    dataset = load_dataset(dataset_name)

    train = dataset["train"]
    base = dataset["base"]
    query = dataset["query"]
    gt = dataset["gt"]

#     save_groundtruth(base, query, "./datasets/deep/deep_groundtruth.ivecs")
#     save_groundtruth(base, query, "./datasets/gist/gist_groundtruth.ivecs")

    os.makedirs(f"./results/{arg}", exist_ok=True)

    class VerboseWriter:
        def __init__(self, *writers):
            self.writers = writers

        def write(self, message):
            for w in self.writers:
                w.write(message)
                w.flush()
        def flush(self):
            for w in self.writers:
                w.flush()

    if dataset_name == "glove":
        m = 30
    else:
        m = 16

    # print(f"[Dataset: {dataset_name}]")
    # print(f"📌 Vector dimension: {train.shape[1]}")
    # print(f"📌 # of subvectors: {m}")
    # print("-" * 60)

    sys.stdout = VerboseWriter(sys.__stdout__, open(f"./results/{arg}/log.txt", "w"))

    if arg == "gist_m":
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_4.csv", train, base, query[:100,:], gt, "k-means", 4)
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_4.csv", train, base, query[:100,:], gt, "k-means", 4)
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_4.csv", train, base, query[:100,:], gt, "k-means", 4)

        print("[m=4]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_4.csv", train, base, query[:100,:], gt, "k-means", 4)
        print("[m=8]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_8.csv", train, base, query[:100,:], gt, "k-means", 8)
        print("[m=16]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_16.csv", train, base, query[:100,:], gt, "k-means", 16)
        print("[m=30]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_30.csv", train, base, query[:100,:], gt, "k-means", 30)
        print("[m=60]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_60.csv", train, base, query[:100,:], gt, "k-means", 60)
        print("[m=120]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_120.csv", train, base, query[:100,:], gt, "k-means", 120)
        print("[m=240]")
        evaluate(pq_result, f"./results/{arg}/k-means_pq_m_240.csv", train, base, query[:100,:], gt, "k-means", 240)
        exit(0)

    evaluate(faiss_pq_result, f"./results/{dataset_name}/faiss_pq_result.csv", train, base, query[:100,:], gt, m)
    evaluate(faiss_opq_result, f"./results/{dataset_name}/faiss_opq_result.csv", train, base, query[:100,:], gt, m)


    evaluate(pq_result, f"./results/{dataset_name}/k-means_pq_result.csv", train, base, query[:100,:], gt, "k-means", m)
    evaluate(pq_result, f"./results/{dataset_name}/k-means_pq_result.csv", train, base, query[:100,:], gt, "k-means", m)
    evaluate(apq_result, f"./results/{dataset_name}/k-means_apq_result.csv", train, base, query[:100,:], gt, "k-means", m)
    evaluate(opq_result, f"./results/{dataset_name}/k-means_opq_result.csv", train, base, query[:100,:], gt, "k-means", m)


    evaluate(pq_result, f"./results/{dataset_name}/k-means++_pq_result.csv", train, base, query[:100,:], gt, "k-means++", m)
    evaluate(pq_result, f"./results/{dataset_name}/k-means++_pq_result.csv", train, base, query[:100,:], gt, "k-means++", m)
    evaluate(apq_result, f"./results/{dataset_name}/k-means++_apq_result.csv", train, base, query[:100,:], gt, "k-means++", m)
    evaluate(opq_result, f"./results/{dataset_name}/k-means++_opq_result.csv", train, base, query[:100,:], gt, "k-means++", m)


    evaluate(pq_result, f"./results/{dataset_name}/mini-batch-k-menas_pq_result.csv", train, base, query[:100,:], gt, "mini-batch-k-means", m)
    evaluate(pq_result, f"./results/{dataset_name}/mini-batch-k-menas_pq_result.csv", train, base, query[:100,:], gt, "mini-batch-k-means", m)
    evaluate(apq_result, f"./results/{dataset_name}/mini-batch-k-menas_apq_result.csv", train, base, query[:100,:], gt, "mini-batch-k-means", m)
    evaluate(opq_result, f"./results/{dataset_name}/mini-batch-k-menas_opq_result.csv", train, base, query[:100,:], gt, "mini-batch-k-means", m)


    evaluate(pq_result, f"./results/{dataset_name}/bisecting-k-means_pq_result.csv", train, base, query[:100,:], gt, "bisecting-k-means", m)
    evaluate(pq_result, f"./results/{dataset_name}/bisecting-k-means_pq_result.csv", train, base, query[:100,:], gt, "bisecting-k-means", m)
    evaluate(apq_result, f"./results/{dataset_name}/bisecting-k-means_apq_result.csv", train, base, query[:100,:], gt, "bisecting-k-means", m)
    evaluate(opq_result, f"./results/{dataset_name}/bisecting-k-means_opq_result.csv", train, base, query[:100,:], gt, "bisecting-k-means", m)


    evaluate(exact_result, f"./results/{dataset_name}/brute_result.csv", base, query[:100,:], gt)


