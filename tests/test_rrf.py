import sys
from pathlib import Path

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ÖNEMLİ:
# BGE-M3 ÖNCE IMPORT EDİLİYOR
# ============================================================

print("=" * 80)
print("RRF TESTI BASLIYOR")
print("=" * 80)

print("\n[1] BGE-M3 import ediliyor...")

from src.embedding_model import BGEEmbeddingModel

print("[OK] BGE-M3 import edildi.")


print("\n[2] BGE-M3 modeli yükleniyor...")

model = BGEEmbeddingModel()

print("[OK] BGE-M3 modeli hazır.")


# ============================================================
# DİĞER KÜTÜPHANELER
# ============================================================

print("\n[3] FAISS import ediliyor...")

import faiss

print("[OK] FAISS import edildi.")


print("\n[4] BM25 import ediliyor...")

from rank_bm25 import BM25Okapi

print("[OK] BM25 import edildi.")


# ============================================================
# AYARLAR
# ============================================================

STRATEGY = "500_100"

TOP_K = 20

RRF_K = 60

QUESTION = (
    "Türkiye’nin Ulusal Yapay Zeka Stratejisi’nin "
    "temel amaçları nelerdir?"
)


# ============================================================
# DOSYA YOLLARI
# ============================================================

metadata_path = (
    PROJECT_ROOT
    / "vector_db"
    / f"metadata_bge_m3_{STRATEGY}.json"
)

index_path = (
    PROJECT_ROOT
    / "vector_db"
    / f"faiss_bge_m3_{STRATEGY}.index"
)


# ============================================================
# METADATA
# ============================================================

print("\n[5] Metadata yükleniyor...")

with open(
    metadata_path,
    "r",
    encoding="utf-8",
) as file:

    metadata = __import__("json").load(file)


print(
    f"[OK] Metadata: {len(metadata)} chunk"
)


# ============================================================
# FAISS
# ============================================================

print("\n[6] FAISS index yükleniyor...")

index = faiss.read_index(
    str(index_path)
)

print(
    f"[OK] Vectors : {index.ntotal}"
)

print(
    f"[OK] Dimension: {index.d}"
)


# ============================================================
# BM25
# ============================================================

print("\n[7] BM25 hazırlanıyor...")


import re


def tokenize(text):

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


corpus = [
    tokenize(chunk["text"])
    for chunk in metadata
]


bm25 = BM25Okapi(corpus)

bm25_scores = bm25.get_scores(
    tokenize(QUESTION)
)


bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True,
)[:TOP_K]


print(
    f"[OK] BM25 Top-K: {len(bm25_indices)}"
)


# ============================================================
# BGE-M3 SEMANTIC SEARCH
# ============================================================

print("\n[8] BGE-M3 soru embedding'i oluşturuluyor...")

query_embedding = model.encode(
    [QUESTION]
)

print(
    f"[OK] Embedding shape: "
    f"{query_embedding.shape}"
)


print("\n[9] FAISS semantic search...")

semantic_scores, semantic_indices = index.search(
    query_embedding,
    TOP_K,
)


semantic_indices = (
    semantic_indices[0].tolist()
)

semantic_scores = (
    semantic_scores[0].tolist()
)


print(
    f"[OK] Semantic Top-K: "
    f"{len(semantic_indices)}"
)


# ============================================================
# RRF
# ============================================================

print("\n[10] RRF hesaplanıyor...")


rrf_scores = {}


# -----------------------------
# Semantic sonuçları
# -----------------------------

for rank, index_id in enumerate(
    semantic_indices,
    start=1,
):

    score = 1 / (
        RRF_K + rank
    )

    rrf_scores[index_id] = (
        rrf_scores.get(
            index_id,
            0,
        )
        + score
    )


# -----------------------------
# BM25 sonuçları
# -----------------------------

for rank, index_id in enumerate(
    bm25_indices,
    start=1,
):

    score = 1 / (
        RRF_K + rank
    )

    rrf_scores[index_id] = (
        rrf_scores.get(
            index_id,
            0,
        )
        + score
    )


# -----------------------------
# Final sıralama
# -----------------------------

final_indices = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True,
)[:10]


# ============================================================
# SONUÇLAR
# ============================================================

print("\n")
print("=" * 80)
print("RRF HYBRID SEARCH SONUCLARI")
print("=" * 80)


for rank, index_id in enumerate(
    final_indices,
    start=1,
):

    print(
        f"\n{rank}. CHUNK {index_id}"
    )

    print(
        f"RRF Score: "
        f"{rrf_scores[index_id]:.6f}"
    )

    print(
        "-" * 80
    )

    print(
        metadata[index_id]["text"][:700]
    )


print("\n")
print("=" * 80)
print("RRF TESTI BASARIYLA TAMAMLANDI")
print("=" * 80)