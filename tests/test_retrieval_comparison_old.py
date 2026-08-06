"""
Semantic Search ve BM25 Retrieval Karşılaştırması

20. Gün Benchmark Testi
"""

import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

CHUNK_STRATEGY = "300_50"

TOP_K = 1

# --------------------------------------------------
# PATHLER
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



# --------------------------------------------------
# TEST SORULARI
# --------------------------------------------------

TEST_QUERIES = [

    {
        "question": "Türkiye'nin çevre sorunları nelerdir?",
        "expected": "cevre_bakanlik",
    },

    {
        "question": "Milli Eğitim Bakanlığının 2024 faaliyetleri nelerdir?",
        "expected": "egitim_meb",
    },

    {
        "question": "Enerji verimliliği neden önemlidir?",
        "expected": "enerji_etkb",
    },

    {
        "question": "Gıda okuryazarlığı nedir?",
        "expected": "gida_tarimorman",
    },

    {
        "question": "Siber güvenlik nedir?",
        "expected": "siber_guvenlik",
    },

    {
        "question": "İklim değişikliğinin tarıma etkileri nelerdir?",
        "expected": "tarim_bakanlik",
    },

    {
        "question": "Tüketici haklarının amacı nedir?",
        "expected": "tuketici_ticaretbakanligi",
    },

    {
        "question": "Türkiye'nin uzay çalışmaları hangi kurum tarafından yürütülmektedir?",
        "expected": "uzay_tubitak",
    },

    {
        "question": "Ulusal Yapay Zeka Stratejisinin amacı nedir?",
        "expected": "yapayzeka",
    },

    {
        "question": "Türk Dil Kurumunun faaliyetleri nelerdir?",
        "expected": "dil_tdk",
    },

]

print("=" * 70)
print("RETRIEVAL KARŞILAŞTIRMA TESTİ")
print("=" * 70)

print(f"\nModel            : {MODEL_NAME}")
print(f"Chunk Stratejisi : {CHUNK_STRATEGY}")
print(f"Toplam Test      : {len(TEST_QUERIES)}")

# --------------------------------------------------
# METADATA
# --------------------------------------------------

print("\nMetadata yükleniyor...")

with METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

print(f"Toplam metadata : {len(metadata)}")


# --------------------------------------------------
# FAISS INDEX
# --------------------------------------------------

print("\nFAISS index yükleniyor...")

with INDEX_PATH.open("rb") as file:
    serialized = np.frombuffer(
        file.read(),
        dtype="uint8",
    )

faiss_index = faiss.deserialize_index(
    serialized
)

print(
    f"Toplam FAISS vektörü : "
    f"{faiss_index.ntotal}"
)


# --------------------------------------------------
# EMBEDDING MODELİ
# --------------------------------------------------

print("\nEmbedding modeli yükleniyor...")

start = time.perf_counter()

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

model_time = (
    time.perf_counter()
    - start
)

print(
    f"Model hazır "
    f"({model_time:.2f} sn)"
)


# --------------------------------------------------
# BM25 TOKENIZER
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

start = time.perf_counter()

bm25 = BM25Okapi(corpus)

bm25_time = (
    time.perf_counter()
    - start
)

print(
    f"BM25 hazır "
    f"({bm25_time:.4f} sn)"
)

# --------------------------------------------------
# FAISS SEARCH
# --------------------------------------------------

def semantic_search(question):
    """
    FAISS üzerinde Top-1 semantic search yapar.
    """

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    start = time.perf_counter()

    scores, indices = faiss_index.search(
        query_embedding,
        TOP_K,
    )

    elapsed = time.perf_counter() - start

    index = indices[0][0]

    return (
        metadata[index],
        float(scores[0][0]),
        elapsed,
    )


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(question):
    """
    BM25 üzerinde Top-1 retrieval yapar.
    """

    query_tokens = tokenize(question)

    start = time.perf_counter()

    scores = bm25.get_scores(query_tokens)

    elapsed = time.perf_counter() - start

    best_index = int(np.argmax(scores))

    return (
        metadata[best_index],
        float(scores[best_index]),
        elapsed,
    )

# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

RRF_K = 60


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):
    """Semantic ve BM25 sonuçlarını RRF ile birleştirir."""

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

    return scores

# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(
    model,
    index,
    bm25,
    metadata,
    question,
):
    """Hybrid Search (RRF)."""

    semantic_results = semantic_search(
        model,
        index,
        question,
    )

    bm25_results = bm25_search(
        bm25,
        question,
    )

    scores = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    idx, score = ranked[0]

    return (
        metadata[idx],
        score,
    )

# --------------------------------------------------
# DOĞRULUK KONTROLÜ
# --------------------------------------------------

def is_correct(result, expected):
    """
    Dönen kaynağın beklenen belge olup olmadığını kontrol eder.
    """

    return expected in result["source"]

# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

benchmark_results = []

semantic_correct = 0
bm25_correct = 0
hybrid_correct = 0

semantic_times = []
bm25_times = []
hybrid_times = []

print("\n" + "=" * 70)
print("BENCHMARK BAŞLIYOR")
print("=" * 70)

for i, test in enumerate(TEST_QUERIES, start=1):

    question = test["question"]
    expected = test["expected"]

    print(f"\n{i}. SORU")
    print(f"Soru     : {question}")
    print(f"Beklenen : {expected}")

    # -------------------------
    # Semantic Search
    # -------------------------

    semantic_result, semantic_score, semantic_time = semantic_search(question)

    semantic_times.append(semantic_time)

    semantic_ok = is_correct(
        semantic_result,
        expected,
    )

    if semantic_ok:
        semantic_correct += 1

    # -------------------------
    # BM25
    # -------------------------

    bm25_result, bm25_score, bm25_time = bm25_search(question)

    bm25_times.append(bm25_time)

    bm25_ok = is_correct(
        bm25_result,
        expected,
    )

    if bm25_ok:
        bm25_correct += 1

    print("\nSemantic Search")
    print("----------------------------")
    print(f"Kaynak : {semantic_result['source']}")
    print(f"Skor   : {semantic_score:.4f}")
    print(f"Süre   : {semantic_time:.4f} sn")
    print(f"Doğru  : {'EVET' if semantic_ok else 'HAYIR'}")

    print("\nBM25")
    print("----------------------------")
    print(f"Kaynak : {bm25_result['source']}")
    print(f"Skor   : {bm25_score:.4f}")
    print(f"Süre   : {bm25_time:.4f} sn")
    print(f"Doğru  : {'EVET' if bm25_ok else 'HAYIR'}")

    benchmark_results.append(
    {
        "question": question,
        "expected": expected,

        "semantic_source": semantic_result["source"],
        "semantic_score": round(semantic_score, 4),
        "semantic_time": round(semantic_time, 4),
        "semantic_correct": semantic_ok,

        "bm25_source": bm25_result["source"],
        "bm25_score": round(bm25_score, 4),
        "bm25_time": round(bm25_time, 4),
        "bm25_correct": bm25_ok,
    }
)

# --------------------------------------------------
# SONUÇLAR
# --------------------------------------------------

print("\n" + "=" * 70)
print("BENCHMARK SONUCU")
print("=" * 70)

semantic_accuracy = (
    semantic_correct / len(TEST_QUERIES)
) * 100

bm25_accuracy = (
    bm25_correct / len(TEST_QUERIES)
) * 100

print(
    f"\nSemantic Accuracy : "
    f"{semantic_correct}/{len(TEST_QUERIES)} "
    f"({semantic_accuracy:.2f}%)"
)

print(
    f"BM25 Accuracy     : "
    f"{bm25_correct}/{len(TEST_QUERIES)} "
    f"({bm25_accuracy:.2f}%)"
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

benchmark_results.append(
    {
        "question": question,
        "expected": expected,

        "semantic_source": semantic_result["source"],
        "semantic_score": round(semantic_score, 4),
        "semantic_time": round(semantic_time, 4),
        "semantic_correct": semantic_ok,

        "bm25_source": bm25_result["source"],
        "bm25_score": round(bm25_score, 4),
        "bm25_time": round(bm25_time, 4),
        "bm25_correct": bm25_ok,
    }
)

print("\n" + "=" * 70)

