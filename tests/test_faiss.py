"""FAISS temel vektör arama testleri."""

# pylint: disable=no-value-for-parameter

import faiss
import numpy as np


# 3 boyutlu örnek vektörler
vectors = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype="float32",
)

# Vektör boyutunu belirle
dimension = vectors.shape[1]

print(f"Vektör boyutu: {dimension}")
print(f"Toplam vektör sayısı: {len(vectors)}")

# L2 (Euclidean distance) kullanan FAISS index
index = faiss.IndexFlatL2(dimension)

# Vektörleri index'e ekle
index.add(vectors)

print(f"FAISS içindeki vektör sayısı: {index.ntotal}")

# Arama sorgusu
query = np.array(
    [
        [1.0, 0.1, 0.0],
    ],
    dtype="float32",
)

# En yakın 2 vektörü getir
TOP_K = 2

distances, indices = index.search(query, TOP_K)

print("\nEn yakın vektörlerin indeksleri:")
print(indices)

print("\nUzaklık değerleri:")
print(distances)