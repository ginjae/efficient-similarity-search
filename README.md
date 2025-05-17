# CSE304 Term Project
20211061 Jaemin Kim

# Efficient Similarity Search via Product Quantization with Improved Clustering

## Quick Start
```bash
conda create --name ess_env
conda activate ess_env
conda install -c pytorch -c nvidia faiss-gpu=1.7.0 pytorch=*=*cuda* pytorch-cuda=11 numpy psutil scikit-learn
python eval.py > ./results/result.txt
```

## Clustering Algorithms
### 1. K-Means
### 2. K-Means++
### 3. Mini-Batch K-Means
### 4. Bisecting K-Means