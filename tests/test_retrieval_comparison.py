"""
RAG Retrieval Benchmark

Karşılaştırılan yöntemler:

1. Semantic Search
2. BM25
3. Hybrid Search (RRF)
4. Hybrid Search + BGE Reranker

Değerlendirme metrikleri:

- Top-1 Accuracy
- Recall@5
- MRR@5
- Search Time
- Reranker ranking change
"""

import csv
import json
import re
import time
from pathlib import Path


# ==========================================================
# KRİTİK IMPORT SIRASI
# ==========================================================
#
# BGE-M3 önce yükleniyor.
#
# Bunun nedeni:
# sentence-transformers / sklearn / pandas tarafındaki
# native DLL'lerin FAISS gibi native kütüphanelerle
# aynı import sırasında çakışmasını önlemek.
#
# search_hybrid.py üzerinde bu sıra ile BGE-M3 başarıyla
# CUDA üzerinde çalıştı.
# ==========================================================

from src.embedding_model import BGEEmbeddingModel


# ==========================================================
# NATIVE KÜTÜPHANELER
# ==========================================================

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_NAME = "BAAI/bge-m3"

CHUNK_STRATEGY = "500_100"

TOP_K = 5

RRF_K = 60

# RRF için kullanılacak aday sayısı.
# Semantic ve BM25 ayrı ayrı 15 aday üretir.
CANDIDATE_K = 15


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

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


# ==========================================================
# TEST QUESTIONS
# ==========================================================

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


# ==========================================================
# BAŞLANGIÇ
# ==========================================================

print("=" * 70)
print("RAG RETRIEVAL BENCHMARK")
print("=" * 70)

print(
    f"\nModel            : {MODEL_NAME}"
)

print(
    f"Chunk Stratejisi : {CHUNK_STRATEGY}"
)

print(
    f"Top-K            : {TOP_K}"
)

print(
    f"RRF K            : {RRF_K}"
)

print(
    f"Aday K           : {CANDIDATE_K}"
)


# ==========================================================
# DOSYA KONTROLLERİ
# ==========================================================

if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Metadata dosyası bulunamadı:\n"
        f"{METADATA_PATH}"
    )


if not INDEX_PATH.exists():

    raise FileNotFoundError(
        f"FAISS index dosyası bulunamadı:\n"
        f"{INDEX_PATH}"
    )


# ==========================================================
# METADATA
# ==========================================================

print("\nMetadata yükleniyor...")

with METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)

print(
    f"Toplam chunk : {len(metadata)}"
)


# ==========================================================
# FAISS
# ==========================================================

print("\nFAISS yükleniyor...")

with INDEX_PATH.open(
    "rb"
) as file:

    serialized = np.frombuffer(
        file.read(),
        dtype="uint8",
    )

faiss_index = faiss.deserialize_index(
    serialized
)

print(
    f"Toplam vektör : "
    f"{faiss_index.ntotal}"
)

print(
    f"Vektör boyutu : "
    f"{faiss_index.d}"
)


# ==========================================================
# KONTROL
# ==========================================================

if faiss_index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index ve metadata "
        "sayıları eşleşmiyor!"
    )


# ==========================================================
# BGE-M3
# ==========================================================

print("\nEmbedding modeli yükleniyor...")

model_start = time.perf_counter()

model = BGEEmbeddingModel()

model_load_time = (
    time.perf_counter()
    - model_start
)

print(
    f"Model hazır "
    f"({model_load_time:.2f} sn)"
)


# ==========================================================
# TOKENIZER
# ==========================================================

def tokenize(text):
    """
    Basit Unicode uyumlu tokenizer.

    Türkçe karakterleri destekler.
    """

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


# ==========================================================
# BM25
# ==========================================================

print("\nBM25 index oluşturuluyor...")

bm25_start = time.perf_counter()

corpus = [
    tokenize(chunk["text"])
    for chunk in metadata
]

bm25 = BM25Okapi(
    corpus
)

bm25_build_time = (
    time.perf_counter()
    - bm25_start
)

print(
    f"BM25 hazır "
    f"({bm25_build_time:.4f} sn)"
)


# ==========================================================
# SOURCE MATCH
# ==========================================================

def source_matches(
    expected,
    metadata_item,
):
    """
    Beklenen kaynak ile metadata kaynağını
    karşılaştırır.
    """

    source = str(
        metadata_item.get(
            "source",
            metadata_item.get(
                "filename",
                metadata_item.get(
                    "file_name",
                    "",
                ),
            ),
        )
    ).lower()

    expected = str(
        expected
    ).lower()

    return (
        source == expected
        or source.startswith(expected)
        or expected in source
    )


# ==========================================================
# RECALL@K
# ==========================================================

def recall_at_k(
    results,
    expected,
    k=TOP_K,
):
    """
    Beklenen kaynak Top-K içerisinde
    bulunuyorsa 1.0 döndürür.
    """

    top_results = results[:k]

    for idx, _ in top_results:

        if source_matches(
            expected,
            metadata[idx],
        ):

            return 1.0

    return 0.0


# ==========================================================
# MRR@K
# ==========================================================

def mrr_at_k(
    results,
    expected,
    k=TOP_K,
):
    """
    Beklenen kaynağın ilk bulunduğu
    sıranın reciprocal değerini döndürür.
    """

    top_results = results[:k]

    for rank, (idx, _) in enumerate(
        top_results,
        start=1,
    ):

        if source_matches(
            expected,
            metadata[idx],
        ):

            return 1.0 / rank

    return 0.0


# ==========================================================
# TOP-1
# ==========================================================

def top1_correct(
    results,
    expected,
):
    """
    İlk sonuç beklenen kaynaktan mı?
    """

    if not results:
        return False

    idx = results[0][0]

    return source_matches(
        expected,
        metadata[idx],
    )


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

def semantic_search(
    question,
    top_k=TOP_K,
):
    """
    BGE-M3 + FAISS Semantic Search.

    Süre:
        embedding + FAISS search
    """

    embedding_start = time.perf_counter()

    query_embedding = model.encode(
        [question],
        batch_size=1,
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    search_start = time.perf_counter()

    scores, indices = faiss_index.search(
        query_embedding,
        top_k,
    )

    search_time = (
        time.perf_counter()
        - search_start
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0],
    ):

        if idx < 0:
            continue

        results.append(
            (
                int(idx),
                float(score),
            )
        )

    total_time = (
        embedding_time
        + search_time
    )

    return (
        results,
        total_time,
    )


# ==========================================================
# BM25 SEARCH
# ==========================================================

def bm25_search(
    question,
    top_k=TOP_K,
):
    """
    BM25 retrieval.
    """

    query_tokens = tokenize(
        question
    )

    start = time.perf_counter()

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    elapsed = (
        time.perf_counter()
        - start
    )

    results = [
        (
            int(idx),
            float(score),
        )
        for idx, score in ranked
    ]

    return (
        results,
        elapsed,
    )


# ==========================================================
# RRF
# ==========================================================

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):
    """
    Semantic + BM25 sonuçlarını
    Reciprocal Rank Fusion ile birleştirir.
    """

    scores = {}

    # ------------------------------------------------------
    # Semantic
    # ------------------------------------------------------

    for rank, (idx, _) in enumerate(
        semantic_results,
        start=1,
    ):

        scores[idx] = (
            scores.get(idx, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # ------------------------------------------------------
    # BM25
    # ------------------------------------------------------

    for rank, (idx, _) in enumerate(
        bm25_results,
        start=1,
    ):

        scores[idx] = (
            scores.get(idx, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# ==========================================================
# HYBRID SEARCH
# ==========================================================

def hybrid_search(
    question,
    top_k=TOP_K,
):
    """
    Semantic + BM25 + RRF.

    Semantic ve BM25 CANDIDATE_K kadar
    aday üretir.
    """

    semantic_results, semantic_time = (
        semantic_search(
            question,
            CANDIDATE_K,
        )
    )

    bm25_results, bm25_time = (
        bm25_search(
            question,
            CANDIDATE_K,
        )
    )

    ranked = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    return (
        ranked[:top_k],
        semantic_results,
        bm25_results,
        semantic_time,
        bm25_time,
    )


# ==========================================================
# HYBRID + RERANKER
# ==========================================================

def hybrid_reranker_search(
    question,
    top_k=TOP_K,
):
    """
    Hybrid + BGE Reranker.

    Önce Hybrid/RRF ile adaylar belirlenir,
    ardından BGE Reranker ile yeniden sıralanır.
    """

    (
        hybrid_results,
        semantic_results,
        bm25_results,
        semantic_time,
        bm25_time,
    ) = hybrid_search(
        question,
        top_k=top_k,
    )

    documents = [
        metadata[idx]["text"]
        for idx, _ in hybrid_results
    ]

    rerank_start = time.perf_counter()

# Reranker burada, ihtiyaç duyulduğu anda yüklenir.
    from src.reranker import rerank

    reranked = rerank(
        question,
        documents,
)

    rerank_time = (
        time.perf_counter()
        - rerank_start
    )

    final_results = []

    for doc_index, rerank_score in (
        reranked[:top_k]
    ):

        original_index = (
            hybrid_results[doc_index][0]
        )

        final_results.append(
            (
                original_index,
                float(rerank_score),
            )
        )

    hybrid_time = (
        semantic_time
        + bm25_time
    )

    total_time = (
        hybrid_time
        + rerank_time
    )

    return (
        final_results,
        hybrid_results,
        semantic_results,
        bm25_results,
        semantic_time,
        bm25_time,
        rerank_time,
        total_time,
    )


# ==========================================================
# BENCHMARK STORAGE
# ==========================================================

benchmark_rows = []


# ----------------------------------------------------------
# TOP-1
# ----------------------------------------------------------

semantic_top1_correct = 0
bm25_top1_correct = 0
hybrid_top1_correct = 0
reranker_top1_correct = 0


# ----------------------------------------------------------
# RECALL
# ----------------------------------------------------------

semantic_recall = []
bm25_recall = []
hybrid_recall = []
reranker_recall = []


# ----------------------------------------------------------
# MRR
# ----------------------------------------------------------

semantic_mrr = []
bm25_mrr = []
hybrid_mrr = []
reranker_mrr = []


# ----------------------------------------------------------
# TIMES
# ----------------------------------------------------------

semantic_times = []
bm25_times = []
hybrid_times = []
reranker_times = []
reranker_total_times = []


# ----------------------------------------------------------
# RERANKER CHANGE
# ----------------------------------------------------------

reranker_changed = 0


# ==========================================================
# BENCHMARK BAŞLIYOR
# ==========================================================

print("\n" + "=" * 70)
print("BENCHMARK BAŞLIYOR")
print("=" * 70)


for question_no, (
    question,
    expected,
) in enumerate(
    TEST_QUERIES,
    start=1,
):

    print("\n" + "-" * 70)

    print(
        f"{question_no}/{len(TEST_QUERIES)}"
    )

    print(
        f"Soru: {question}"
    )

    print(
        f"Beklenen kaynak: {expected}"
    )

    # ======================================================
    # SEMANTIC
    # ======================================================

    semantic_results, semantic_time = (
        semantic_search(
            question,
            TOP_K,
        )
    )

    semantic_times.append(
        semantic_time
    )

    # ======================================================
    # BM25
    # ======================================================

    bm25_results, bm25_time = (
        bm25_search(
            question,
            TOP_K,
        )
    )

    bm25_times.append(
        bm25_time
    )

    # ======================================================
    # HYBRID
    # ======================================================

    (
        hybrid_results,
        _,
        _,
        hybrid_semantic_time,
        hybrid_bm25_time,
    ) = hybrid_search(
        question,
        TOP_K,
    )

    hybrid_time = (
        hybrid_semantic_time
        + hybrid_bm25_time
    )

    hybrid_times.append(
        hybrid_time
    )

    # ======================================================
    # RERANKER
    # ======================================================

    (
        reranker_results,
        original_hybrid_results,
        _,
        _,
        _,
        _,
        rerank_time,
        reranker_total_time,
    ) = hybrid_reranker_search(
        question,
        TOP_K,
    )

    reranker_times.append(
        rerank_time
    )

    reranker_total_times.append(
        reranker_total_time
    )

    # ======================================================
    # TOP-1
    # ======================================================

    semantic_ok = top1_correct(
        semantic_results,
        expected,
    )

    bm25_ok = top1_correct(
        bm25_results,
        expected,
    )

    hybrid_ok = top1_correct(
        hybrid_results,
        expected,
    )

    reranker_ok = top1_correct(
        reranker_results,
        expected,
    )

    semantic_top1_correct += int(
        semantic_ok
    )

    bm25_top1_correct += int(
        bm25_ok
    )

    hybrid_top1_correct += int(
        hybrid_ok
    )

    reranker_top1_correct += int(
        reranker_ok
    )

    # ======================================================
    # RECALL@5
    # ======================================================

    semantic_r5 = recall_at_k(
        semantic_results,
        expected,
        TOP_K,
    )

    bm25_r5 = recall_at_k(
        bm25_results,
        expected,
        TOP_K,
    )

    hybrid_r5 = recall_at_k(
        hybrid_results,
        expected,
        TOP_K,
    )

    reranker_r5 = recall_at_k(
        reranker_results,
        expected,
        TOP_K,
    )

    semantic_recall.append(
        semantic_r5
    )

    bm25_recall.append(
        bm25_r5
    )

    hybrid_recall.append(
        hybrid_r5
    )

    reranker_recall.append(
        reranker_r5
    )

    # ======================================================
    # MRR@5
    # ======================================================

    semantic_m = mrr_at_k(
        semantic_results,
        expected,
        TOP_K,
    )

    bm25_m = mrr_at_k(
        bm25_results,
        expected,
        TOP_K,
    )

    hybrid_m = mrr_at_k(
        hybrid_results,
        expected,
        TOP_K,
    )

    reranker_m = mrr_at_k(
        reranker_results,
        expected,
        TOP_K,
    )

    semantic_mrr.append(
        semantic_m
    )

    bm25_mrr.append(
        bm25_m
    )

    hybrid_mrr.append(
        hybrid_m
    )

    reranker_mrr.append(
        reranker_m
    )

    # ======================================================
    # RERANKER DEĞİŞİMİ
    # ======================================================

    original_order = [
        idx
        for idx, _ in original_hybrid_results
    ]

    reranked_order = [
        idx
        for idx, _ in reranker_results
    ]

    changed = (
        original_order != reranked_order
    )

    reranker_changed += int(
        changed
    )

    # ======================================================
    # CSV TOP-5 ID'LER
    # ======================================================

    semantic_top5 = "|".join(
        str(idx)
        for idx, _ in semantic_results
    )

    bm25_top5 = "|".join(
        str(idx)
        for idx, _ in bm25_results
    )

    hybrid_top5 = "|".join(
        str(idx)
        for idx, _ in hybrid_results
    )

    reranker_top5 = "|".join(
        str(idx)
        for idx, _ in reranker_results
    )

    # ======================================================
    # CSV ROW
    # ======================================================

    benchmark_rows.append([

        question,
        expected,

        # --------------------------------------------------
        # Semantic
        # --------------------------------------------------

        semantic_results[0][0]
        if semantic_results
        else "",

        semantic_results[0][1]
        if semantic_results
        else "",

        semantic_time,

        semantic_ok,

        # --------------------------------------------------
        # BM25
        # --------------------------------------------------

        bm25_results[0][0]
        if bm25_results
        else "",

        bm25_results[0][1]
        if bm25_results
        else "",

        bm25_time,

        bm25_ok,

        # --------------------------------------------------
        # Hybrid
        # --------------------------------------------------

        hybrid_results[0][0]
        if hybrid_results
        else "",

        hybrid_results[0][1]
        if hybrid_results
        else "",

        hybrid_time,

        hybrid_ok,

        # --------------------------------------------------
        # Reranker
        # --------------------------------------------------

        reranker_results[0][0]
        if reranker_results
        else "",

        reranker_results[0][1]
        if reranker_results
        else "",

        reranker_total_time,

        reranker_ok,

        # --------------------------------------------------
        # Top-5
        # --------------------------------------------------

        semantic_top5,
        bm25_top5,
        hybrid_top5,
        reranker_top5,

        # --------------------------------------------------
        # Recall
        # --------------------------------------------------

        semantic_r5,
        bm25_r5,
        hybrid_r5,
        reranker_r5,

        # --------------------------------------------------
        # MRR
        # --------------------------------------------------

        semantic_m,
        bm25_m,
        hybrid_m,
        reranker_m,

        # --------------------------------------------------
        # Reranker change
        # --------------------------------------------------

        changed,
    ])

    # ======================================================
    # EKRAN
    # ======================================================

    print(
        f"Semantic Top-1 : "
        f"{'EVET' if semantic_ok else 'HAYIR'}"
    )

    print(
        f"BM25 Top-1     : "
        f"{'EVET' if bm25_ok else 'HAYIR'}"
    )

    print(
        f"Hybrid Top-1   : "
        f"{'EVET' if hybrid_ok else 'HAYIR'}"
    )

    print(
        f"Reranker Top-1 : "
        f"{'EVET' if reranker_ok else 'HAYIR'}"
    )

    print(
        f"Recall@5       : "
        f"Semantic={semantic_r5:.0f} "
        f"BM25={bm25_r5:.0f} "
        f"Hybrid={hybrid_r5:.0f} "
        f"Reranker={reranker_r5:.0f}"
    )

    print(
        f"MRR@5          : "
        f"Semantic={semantic_m:.3f} "
        f"BM25={bm25_m:.3f} "
        f"Hybrid={hybrid_m:.3f} "
        f"Reranker={reranker_m:.3f}"
    )

    print(
        f"Süre            : "
        f"Semantic={semantic_time:.4f}s "
        f"BM25={bm25_time:.4f}s "
        f"Hybrid={hybrid_time:.4f}s "
        f"Reranker={rerank_time:.4f}s"
    )


# ==========================================================
# MEAN
# ==========================================================

def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# ==========================================================
# QUESTION COUNT
# ==========================================================

question_count = len(
    TEST_QUERIES
)


# ==========================================================
# ACCURACY
# ==========================================================

semantic_accuracy = (
    semantic_top1_correct
    / question_count
    * 100
)

bm25_accuracy = (
    bm25_top1_correct
    / question_count
    * 100
)

hybrid_accuracy = (
    hybrid_top1_correct
    / question_count
    * 100
)

reranker_accuracy = (
    reranker_top1_correct
    / question_count
    * 100
)


# ==========================================================
# RECALL AVERAGE
# ==========================================================

semantic_recall_avg = mean(
    semantic_recall
)

bm25_recall_avg = mean(
    bm25_recall
)

hybrid_recall_avg = mean(
    hybrid_recall
)

reranker_recall_avg = mean(
    reranker_recall
)


# ==========================================================
# MRR AVERAGE
# ==========================================================

semantic_mrr_avg = mean(
    semantic_mrr
)

bm25_mrr_avg = mean(
    bm25_mrr
)

hybrid_mrr_avg = mean(
    hybrid_mrr
)

reranker_mrr_avg = mean(
    reranker_mrr
)


# ==========================================================
# TIME AVERAGE
# ==========================================================

semantic_time_avg = mean(
    semantic_times
)

bm25_time_avg = mean(
    bm25_times
)

hybrid_time_avg = mean(
    hybrid_times
)

reranker_time_avg = mean(
    reranker_times
)

reranker_total_time_avg = mean(
    reranker_total_times
)


# ==========================================================
# CSV KAYDET
# ==========================================================

print("\n" + "=" * 70)
print("CSV KAYDEDİLİYOR")
print("=" * 70)


with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([

        "question",
        "expected",

        "semantic_source",
        "semantic_score",
        "semantic_time",
        "semantic_correct",

        "bm25_source",
        "bm25_score",
        "bm25_time",
        "bm25_correct",

        "hybrid_source",
        "hybrid_score",
        "hybrid_time",
        "hybrid_correct",

        "reranker_source",
        "reranker_score",
        "reranker_time",
        "reranker_correct",

        "semantic_top5",
        "bm25_top5",
        "hybrid_top5",
        "reranker_top5",

        "semantic_recall_at_5",
        "bm25_recall_at_5",
        "hybrid_recall_at_5",
        "reranker_recall_at_5",

        "semantic_mrr_at_5",
        "bm25_mrr_at_5",
        "hybrid_mrr_at_5",
        "reranker_mrr_at_5",

        "reranker_changed",
    ])

    writer.writerows(
        benchmark_rows
    )


# ==========================================================
# SONUÇLAR
# ==========================================================

print("\n" + "=" * 70)
print("BENCHMARK SONUÇLARI")
print("=" * 70)


# ==========================================================
# TOP-1
# ==========================================================

print("\nTOP-1 DOĞRULUK")
print("-" * 40)

print(
    f"Semantic Search       : "
    f"{semantic_accuracy:.2f}%"
)

print(
    f"BM25                  : "
    f"{bm25_accuracy:.2f}%"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_accuracy:.2f}%"
)

print(
    f"Hybrid + Reranker     : "
    f"{reranker_accuracy:.2f}%"
)


# ==========================================================
# RECALL
# ==========================================================

print("\nRECALL@5")
print("-" * 40)

print(
    f"Semantic Search       : "
    f"{semantic_recall_avg:.3f}"
)

print(
    f"BM25                  : "
    f"{bm25_recall_avg:.3f}"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_recall_avg:.3f}"
)

print(
    f"Hybrid + Reranker     : "
    f"{reranker_recall_avg:.3f}"
)


# ==========================================================
# MRR
# ==========================================================

print("\nMRR@5")
print("-" * 40)

print(
    f"Semantic Search       : "
    f"{semantic_mrr_avg:.3f}"
)

print(
    f"BM25                  : "
    f"{bm25_mrr_avg:.3f}"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_mrr_avg:.3f}"
)

print(
    f"Hybrid + Reranker     : "
    f"{reranker_mrr_avg:.3f}"
)


# ==========================================================
# TIME
# ==========================================================

print("\nORTALAMA SÜRE")
print("-" * 40)

print(
    f"Semantic Search       : "
    f"{semantic_time_avg:.4f} sn"
)

print(
    f"BM25                  : "
    f"{bm25_time_avg:.4f} sn"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_time_avg:.4f} sn"
)

print(
    f"Reranker              : "
    f"{reranker_time_avg:.4f} sn"
)

print(
    f"Hybrid + Reranker     : "
    f"{reranker_total_time_avg:.4f} sn"
)


# ==========================================================
# RERANKER DEĞİŞİMİ
# ==========================================================

print("\nRERANKER DEĞİŞİMİ")
print("-" * 40)

print(
    f"Sıralaması değişen soru : "
    f"{reranker_changed}/{question_count}"
)

print(
    f"Değişim oranı           : "
    f"{reranker_changed / question_count * 100:.2f}%"
)


# ==========================================================
# CSV
# ==========================================================

print("\nCSV:")
print(CSV_PATH)


# ==========================================================
# BİTİŞ
# ==========================================================

print("\n" + "=" * 70)
print("BENCHMARK TAMAMLANDI")
print("=" * 70)