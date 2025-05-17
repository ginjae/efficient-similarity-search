"""

[EVALUATION]

ANN 벤치마크 데이터셋 대상으로 
1. accuracy (recall@k)
2. memory usage
3. search time
4. training time

--------------------------------------------------------------------------------

[QUESTION]

모든 구현에 parallelization을 적용시킬 것인가
ProductQuantizer에서 numpy 대신 pytorch로 바꾸면 GPU 사용이 가능한가


"""

import os
import subprocess
import psutil
import threading
import time
import csv

# PQ
from product_quantizer import ProductQuantizer

# faiss
import faiss
from faiss.contrib.vecs_io import ivecs_read, fvecs_read


def download_dataset(name):
    dest_dir = "./datasets"
    os.makedirs(dest_dir, exist_ok=True)
    target_dir = os.path.join(dest_dir, "sift")
    
    if os.path.exists(target_dir):
        print(f"[INFO] '{target_dir}' 디렉토리가 이미 존재합니다. 다운로드를 생략합니다.")
        return target_dir
    if name.lower() == 'sift1m':
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        archive_path = os.path.join(dest_dir, "sift.tar.gz")

        print(f"[INFO] SIFT1M 데이터셋 다운로드 중... ({url})")
        subprocess.run(f"wget -O {archive_path} {url}", shell=True, check=True)

        print(f"[INFO] 압축 해제 중... ({archive_path})")
        subprocess.run(f"tar -xzf {archive_path} -C {dest_dir}", shell=True, check=True)

        print(f"[INFO] 다운로드 및 압축 해제 완료: {target_dir}")
    else:
        raise ValueError(f"지원되지 않는 데이터셋 이름: '{name}'")
    return target_dir

def load_dataset(name):
    xt = fvecs_read("datasets/sift/sift_learn.fvecs")
    xb = fvecs_read("datasets/sift/sift_base.fvecs")
    xq = fvecs_read("datasets/sift/sift_query.fvecs")
    gt = ivecs_read("datasets/sift/sift_groundtruth.ivecs")

    return { "train": xt, "base": xb, "query": xq, "gt": gt }

def get_hit(a: set, b: set):
    return len(a & b)

def get_recall(a, b, k):
    cumulative_recall = 0
    n = len(a)
    for i in range(n):
        cumulative_recall += get_hit(set(a[i]), set(b[i])) / k;
    return cumulative_recall / n

def brute_result(base, query, gt):
    pq = ProductQuantizer()
    print("brute search start")
    search_start = time.perf_counter();
    _, topk_idx = pq.search_original(query, base, 100)
    search_end = time.perf_counter();
    print(f"brute search time: {search_end - search_start}")
    print(f"[BRUTE] recall: {get_recall(topk_idx, gt, 100)}")

def pq_result(train, base, query, gt, clustering):
    pq = ProductQuantizer(clustering=clustering, M=16, Ks=256)
    print(clustering + " pq training start")
    training_start = time.perf_counter();
    pq.train(train)
    pq.add(base)
    training_end = time.perf_counter();
    print(f"{clustering} pq training time: {training_end - training_start}")

    print(clustering + " pq search start")
    search_start = time.perf_counter();
    _, topk_idx = pq.search(query, topk=100)
    search_end = time.perf_counter();
    print(f"{clustering} pq search time: {search_end - search_start}")
    print(f"[{clustering.upper()} PQ] recall: {get_recall(topk_idx, gt, 100)}")

def faiss_result(train, base, query, gt):
    indexPQ = faiss.IndexPQ(train.shape[1], 16, 8, faiss.METRIC_L2)
    print("faiss training start")
    training_start = time.perf_counter();
    indexPQ.train(train)
    indexPQ.add(base)
    training_end = time.perf_counter();
    print(f"faiss training time: {training_end - training_start}")

    print("faiss search start")
    search_start = time.perf_counter();
    _, topk_idx = indexPQ.search(query, 100)
    search_end = time.perf_counter();
    print(f"faiss search time: {search_end - search_start}")
    print(f"[FAISS] recall: {get_recall(topk_idx, gt, 100)}")

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


if __name__ == "__main__":

    download_dataset("sift1M")
    dataset = load_dataset("sift1M")

    X_train = dataset["train"]
    X_base = dataset["base"]
    X_query = dataset["query"]
    gt = dataset["gt"]

#     faiss_recall = faiss_result(X_train, X_base, X_query, gt)
#     pq_recall = pq_result(X_train, X_base, X_query, gt)
#     brute_recall = brute_result(X_train, X_base, X_query, gt)

    os.makedirs("./results", exist_ok=True)
    evaluate(faiss_result, "./results/faiss_result.csv", X_train, X_base, X_query[:100,:], gt)
    evaluate(pq_result, "./results/k-means_pq_result.csv", X_train, X_base, X_query[:100,:], gt, "k-means")
    evaluate(pq_result, "./results/k-means++_pq_result.csv", X_train, X_base, X_query[:100,:], gt, "k-means++")
    evaluate(pq_result, "./results/mini-batch-k-menas_pq_result.csv", X_train, X_base, X_query[:100,:], gt, "mini-batch-k-means")
    evaluate(pq_result, "./results/bisecting-k-means_pq_result.csv", X_train, X_base, X_query[:100,:], gt, "bisecting-k-means")
    evaluate(brute_result, "./results/brute_result.csv", X_base, X_query[:100,:], gt)

