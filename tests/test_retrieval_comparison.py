"""
Semantic Search - BM25 - Hybrid Search Benchmark

22. Gün
"""

import csv
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

CHUNK_STRATEGY = "300_50"

TOP_K = 5

RRF_K = 60

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "vector_db"
    / f"faiss_bge_m3_{CHUNK_STRATEGY}.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "vector_db"
    / f"metadata_bge_m3_{CHUNK_STRATEGY}.json"
)

CSV_PATH = (
    PROJECT_ROOT
    / "benchmark_results.csv"
)

# --------------------------------------------------
# TEST QUERIES
# --------------------------------------------------

TEST_QUERIES = [

    (
        "Türkiye'nin çevre sorunları nelerdir?",
        "cevre_bakanlik",
    ),

    (
        "Milli Eğitim Bakanlığının 2024 faaliyetleri nelerdir?",
        "egitim_meb",
    ),

    (
        "Enerji verimliliği neden önemlidir?",
        "enerji_etkb",
    ),

    (
        "Gıda okuryazarlığı nedir?",
        "gida_tarimorman",
    ),

    (
        "Siber güvenlik nedir?",
        "siber_guvenlik",
    ),

    (
        "İklim değişikliğinin tarıma etkileri nelerdir?",
        "tarim_bakanlik",
    ),

    (
        "Tüketici haklarının amacı nedir?",
        "tuketici_ticaretbakanligi",
    ),

    (
        "Türkiye'nin uzay çalışmaları hangi kurum tarafından yürütülmektedir?",
        "uzay_tubitak",
    ),

    (
        "Ulusal Yapay Zeka Stratejisinin amacı nedir?",
        "yapayzeka",
    ),

    (
        "Türk Dil Kurumunun faaliyetleri nelerdir?",
        "dil_tdk",
    ),

]

print("=" * 70)
print("SEMANTIC - BM25 - HYBRID BENCHMARK")
print("=" * 70)

print("\nMetadata yükleniyor...")

with METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)

print(f"Toplam chunk : {len(metadata)}")

print("\nFAISS yükleniyor...")

with INDEX_PATH.open("rb") as file:

    serialized = np.frombuffer(
        file.read(),
        dtype="uint8",
    )

faiss_index = faiss.deserialize_index(serialized)

print(f"Toplam vektör : {faiss_index.ntotal}")

print("\nEmbedding modeli yükleniyor...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

print("Model hazır.")

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    """Basit Türkçe tokenizer."""

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


# --------------------------------------------------
# BM25 INDEX
# --------------------------------------------------

print("\nBM25 index oluşturuluyor...")

corpus = [
    tokenize(chunk["text"])
    for chunk in metadata
]

bm25 = BM25Okapi(corpus)

print("BM25 hazır.")


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(question, top_k=TOP_K):
    """FAISS Semantic Search"""

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    start = time.perf_counter()

    scores, indices = faiss_index.search(
        query_embedding,
        top_k,
    )

    elapsed = time.perf_counter() - start

    results = []

    for idx, score in zip(indices[0], scores[0]):

        results.append(
            (
                idx,
                float(score),
            )
        )

    return results, elapsed


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(question, top_k=TOP_K):
    """BM25 Retrieval"""

    query_tokens = tokenize(question)

    start = time.perf_counter()

    scores = bm25.get_scores(query_tokens)

    elapsed = time.perf_counter() - start

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return ranked, elapsed


# --------------------------------------------------
# RRF
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):
    """Reciprocal Rank Fusion"""

    scores = {}

    for rank, (idx, _) in enumerate(
        semantic_results,
        start=1,
    ):

        scores[idx] = (
            scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, (idx, _) in enumerate(
        bm25_results,
        start=1,
    ):

        scores[idx] = (
            scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(question, top_k=TOP_K):
    """Hybrid Search (Semantic + BM25 + RRF)"""

    semantic_results, semantic_time = semantic_search(
        question,
        top_k * 3,
    )

    bm25_results, bm25_time = bm25_search(
        question,
        top_k * 3,
    )

    ranked = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    return (
        ranked[:top_k],
        semantic_time + bm25_time,
    )


# --------------------------------------------------
# DOĞRULUK
# --------------------------------------------------

def is_correct(
    source,
    expected,
):

    return expected in source

# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

print("\n" + "=" * 70)
print("BENCHMARK BAŞLIYOR")
print("=" * 70)

semantic_correct = 0
bm25_correct = 0
hybrid_correct = 0

semantic_times = []
bm25_times = []
hybrid_times = []

benchmark_rows = []

for i, (question, expected) in enumerate(TEST_QUERIES, start=1):

    print(f"\n{i}. SORU")
    print(f"Soru     : {question}")
    print(f"Beklenen : {expected}")

    # --------------------------------------------------
    # Semantic Search
    # --------------------------------------------------

    semantic_results, semantic_time = semantic_search(question)

    semantic_idx, semantic_score = semantic_results[0]

    semantic_source = metadata[semantic_idx]["source"]

    semantic_ok = is_correct(
        semantic_source,
        expected,
    )

    semantic_times.append(semantic_time)

    if semantic_ok:
        semantic_correct += 1

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25_results, bm25_time = bm25_search(question)

    bm25_idx, bm25_score = bm25_results[0]

    bm25_source = metadata[bm25_idx]["source"]

    bm25_ok = is_correct(
        bm25_source,
        expected,
    )

    bm25_times.append(bm25_time)

    if bm25_ok:
        bm25_correct += 1

    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------

    hybrid_results, hybrid_time = hybrid_search(question)

    hybrid_idx, hybrid_score = hybrid_results[0]

    hybrid_source = metadata[hybrid_idx]["source"]

    hybrid_ok = is_correct(
        hybrid_source,
        expected,
    )

    hybrid_times.append(hybrid_time)

    if hybrid_ok:
        hybrid_correct += 1

    print("\nSemantic")
    print("----------------------------")
    print(f"Kaynak : {semantic_source}")
    print(f"Skor   : {semantic_score:.4f}")
    print(f"Doğru  : {'EVET' if semantic_ok else 'HAYIR'}")

    print("\nBM25")
    print("----------------------------")
    print(f"Kaynak : {bm25_source}")
    print(f"Skor   : {bm25_score:.4f}")
    print(f"Doğru  : {'EVET' if bm25_ok else 'HAYIR'}")

    print("\nHybrid")
    print("----------------------------")
    print(f"Kaynak : {hybrid_source}")
    print(f"Skor   : {hybrid_score:.4f}")
    print(f"Doğru  : {'EVET' if hybrid_ok else 'HAYIR'}")

    benchmark_rows.append([
        question,
        expected,
        semantic_source,
        semantic_score,
        semantic_ok,
        bm25_source,
        bm25_score,
        bm25_ok,
        hybrid_source,
        hybrid_score,
        hybrid_ok,
    ])

# --------------------------------------------------
# CSV
# --------------------------------------------------

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "question",
        "expected",
        "semantic_source",
        "semantic_score",
        "semantic_correct",
        "bm25_source",
        "bm25_score",
        "bm25_correct",
        "hybrid_source",
        "hybrid_score",
        "hybrid_correct",
    ])

    writer.writerows(benchmark_rows)

# --------------------------------------------------
# SONUÇLAR
# --------------------------------------------------

print("\n" + "=" * 70)
print("BENCHMARK SONUCU")
print("=" * 70)

print(
    f"\nSemantic Accuracy : "
    f"{semantic_correct}/{len(TEST_QUERIES)} "
    f"({semantic_correct/len(TEST_QUERIES)*100:.2f}%)"
)

print(
    f"BM25 Accuracy     : "
    f"{bm25_correct}/{len(TEST_QUERIES)} "
    f"({bm25_correct/len(TEST_QUERIES)*100:.2f}%)"
)

print(
    f"Hybrid Accuracy   : "
    f"{hybrid_correct}/{len(TEST_QUERIES)} "
    f"({hybrid_correct/len(TEST_QUERIES)*100:.2f}%)"
)

print()

print(
    f"Semantic Ortalama Süre : "
    f"{sum(semantic_times)/len(semantic_times):.4f} sn"
)

print(
    f"BM25 Ortalama Süre     : "
    f"{sum(bm25_times)/len(bm25_times):.4f} sn"
)

print(
    f"Hybrid Ortalama Süre   : "
    f"{sum(hybrid_times)/len(hybrid_times):.4f} sn"
)

print("\nCSV kaydedildi:")
print(CSV_PATH)

print("\n" + "=" * 70)