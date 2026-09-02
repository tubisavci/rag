"""
Hybrid Search (Semantic + BM25 + Reranker)

Kullanım:

python -m src.search_hybrid --strategy 300_50
python -m src.search_hybrid --strategy 500_100
python -m src.search_hybrid --strategy 800_150
"""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch

from src.embedding_model import BGEEmbeddingModel

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5

# Reciprocal Rank Fusion sabiti
RRF_K = 60


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():
    """Komut satırı argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description="Hybrid Search"
    )

    parser.add_argument(
        "--strategy",
        default="300_50",
        choices=[
            "300_50",
            "500_100",
            "800_150",
        ],
        help="Chunk stratejisi",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help="Getirilecek sonuç sayısı",
    )

    return parser.parse_args()


# --------------------------------------------------
# PATHLER
# --------------------------------------------------

def get_paths(strategy):
    """FAISS index ve metadata dosya yollarını oluşturur."""

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    index_path = (
        project_root
        / "vector_db"
        / f"faiss_bge_m3_{strategy}.index"
    )

    metadata_path = (
        project_root
        / "vector_db"
        / f"metadata_bge_m3_{strategy}.json"
    )

    return index_path, metadata_path


# --------------------------------------------------
# FAISS INDEX
# --------------------------------------------------

def load_index(index_path):
    """FAISS indexini yükler."""

    print("\nFAISS index yükleniyor...")

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index bulunamadı: {index_path}"
        )

    import faiss
    import numpy as np

    with index_path.open("rb") as file:
        serialized = np.frombuffer(
            file.read(),
            dtype="uint8",
        )

    index = faiss.deserialize_index(
        serialized
    )

    print(
        f"Toplam vektör : {index.ntotal}"
    )

    print(
        f"Vektör boyutu : {index.d}"
    )

    return index


# --------------------------------------------------
# METADATA
# --------------------------------------------------

def load_metadata(metadata_path):
    """Metadata JSON dosyasını yükler."""

    print("\nMetadata yükleniyor...")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata bulunamadı: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            "Metadata list formatında olmalıdır."
        )

    print(
        f"Toplam metadata : {len(metadata)}"
    )

    return metadata


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

def load_model():
    """BGE-M3 embedding modelini yükler."""

    print("\nEmbedding modeli yükleniyor...")

    start = time.perf_counter()

    model = BGEEmbeddingModel()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Model hazır "
        f"({elapsed:.2f} sn)"
    )

    return (
        model,
        elapsed,
    )


# --------------------------------------------------
# MODEL DEVICE YÖNETİMİ
# --------------------------------------------------

def move_embedding_model_to_gpu(model):
    """
    BGE-M3 modelini GPU'ya taşır.

    Model yeniden yüklenmez.
    Sadece mevcut model CPU'dan GPU'ya alınır.
    """

    if not torch.cuda.is_available():
        return

    if model.device != "cuda":
        print("\nBGE-M3 GPU'ya taşınıyor...")

        start = time.perf_counter()

        model.model.to("cuda")
        model.device = "cuda"

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"BGE-M3 GPU hazır "
            f"({elapsed:.4f} sn)"
        )


def move_embedding_model_to_cpu(model):
    """
    Semantic retrieval tamamlandıktan sonra
    BGE-M3 modelini CPU'ya taşır.

    Böylece reranker için GPU belleği serbest bırakılır.
    """

    if not torch.cuda.is_available():
        return

    if model.device == "cuda":

        print("\nBGE-M3 CPU'ya taşınıyor...")

        start = time.perf_counter()

        model.model.to("cpu")
        model.device = "cpu"

        # Python referanslarını temizle
        gc.collect()

        # CUDA cache temizle
        torch.cuda.empty_cache()

        elapsed = (
            time.perf_counter()
            - start
        )

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        print(
            f"BGE-M3 CPU'da "
            f"({elapsed:.4f} sn)"
        )

        print(
            f"GPU belleği : "
            f"{allocated:.2f} MB"
        )

        print(
            f"GPU reserved: "
            f"{reserved:.2f} MB"
        )


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    """Basit Türkçe kelime tokenizer."""

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


# --------------------------------------------------
# BM25
# --------------------------------------------------

def build_bm25(metadata):
    """Metadata üzerinden BM25 index oluşturur."""

    print("\nBM25 index oluşturuluyor...")

    start = time.perf_counter()

    from rank_bm25 import BM25Okapi

    corpus = [
        tokenize(chunk["text"])
        for chunk in metadata
    ]

    bm25 = BM25Okapi(corpus)

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"BM25 hazır ({elapsed:.4f} sn)"
    )

    return (
        bm25,
        elapsed,
    )


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(
    model,
    index,
    question,
    top_k,
):
    """
    BGE-M3 + FAISS ile semantic search yapar.
    """

    start = time.perf_counter()

    # Yeni sorgu için model GPU'da olmalı.
    move_embedding_model_to_gpu(
        model
    )

    query_embedding = model.encode(
        [question],
        batch_size=1,
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
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

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Semantic Search : "
        f"{elapsed:.4f} sn"
    )

    return results


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(
    bm25,
    question,
    top_k,
):
    """BM25 retrieval gerçekleştirir."""

    start = time.perf_counter()

    query_tokens = tokenize(
        question
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = [
        (
            int(idx),
            float(score),
        )
        for idx, score in ranked
    ]

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"BM25 Search    : "
        f"{elapsed:.4f} sn"
    )

    return results


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):
    """
    Semantic Search ve BM25 sonuçlarını
    Reciprocal Rank Fusion ile birleştirir.
    """

    scores = {}

    # Semantic sonuçları
    for rank, (idx, _) in enumerate(
        semantic_results,
        start=1,
    ):

        idx = int(idx)

        scores[idx] = (
            scores.get(idx, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # BM25 sonuçları
    for rank, (idx, _) in enumerate(
        bm25_results,
        start=1,
    ):

        idx = int(idx)

        scores[idx] = (
            scores.get(idx, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


# --------------------------------------------------
# HYBRID SEARCH + RERANKER
# --------------------------------------------------

def hybrid_search(
    model,
    faiss_index,
    bm25,
    metadata,
    question,
    top_k,
):
    """
    Tam retrieval pipeline:

    1. Semantic Search
    2. BM25
    3. RRF
    4. BGE-M3 GPU -> CPU
    5. Reranker
    """

    print("\n" + "-" * 70)
    print("SEMANTIC SEARCH + BM25")
    print("-" * 70)

    retrieval_k = max(
        top_k * 3,
        15,
    )

    # --------------------------------------------------
    # 1. SEMANTIC SEARCH
    # --------------------------------------------------

    semantic_results = semantic_search(
        model=model,
        index=faiss_index,
        question=question,
        top_k=retrieval_k,
    )

    # --------------------------------------------------
    # 2. BM25
    # --------------------------------------------------

    bm25_results = bm25_search(
        bm25=bm25,
        question=question,
        top_k=retrieval_k,
    )

    # --------------------------------------------------
    # 3. RRF
    # --------------------------------------------------

    rrf_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    reranker_k = max(
        top_k * 2,
        10,
    )

    candidates = rrf_results[
        :reranker_k
    ]

    print(
        f"\nRRF aday sayısı : "
        f"{len(candidates)}"
    )

    print("\nRRF ilk adaylar:")

    for rank, (idx, score) in enumerate(
        candidates[:5],
        start=1,
    ):

        print(
            f"{rank}. Chunk {idx} "
            f"| RRF: {score:.6f}"
        )

    if not candidates:
        return []

    # --------------------------------------------------
    # 4. RERANKER ÖNCESİ GPU TEMİZLİĞİ
    # --------------------------------------------------

    move_embedding_model_to_cpu(
        model
    )

    # --------------------------------------------------
    # 5. RERANKER
    # --------------------------------------------------

    documents = [
        metadata[int(idx)]["text"]
        for idx, _ in candidates
    ]

    print("\nReranker çalıştırılıyor...")

    from src.reranker import rerank

    reranked = rerank(
        question,
        documents,
    )

    # --------------------------------------------------
    # 6. SONUÇLARI EŞLEŞTİR
    # --------------------------------------------------

    final_results = []

    for doc_index, rerank_score in reranked:

        if doc_index >= len(candidates):
            continue

        original_index = int(
            candidates[doc_index][0]
        )

        final_results.append(
            (
                metadata[original_index],
                float(rerank_score),
            )
        )

        if len(final_results) >= top_k:
            break

    return final_results


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

def print_results(
    question,
    results,
):
    """Hybrid Search sonuçlarını yazdırır."""

    print("\n" + "=" * 70)
    print("HYBRID SEARCH SONUCU")
    print("=" * 70)

    print(
        f"\nSoru: {question}"
    )

    print(
        f"\nToplam sonuç: {len(results)}"
    )

    print("\n" + "=" * 70)

    for rank, (
        chunk,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n{rank}. SONUÇ"
        )

        print(
            f"Reranker Skoru : "
            f"{score:.4f}"
        )

        print(
            f"Kaynak         : "
            f"{chunk.get('source', '-')}"
        )

        print(
            f"Chunk ID       : "
            f"{chunk.get('chunk_id', '-')}"
        )

        print(
            f"Chunk Index    : "
            f"{chunk.get('chunk_index', '-')}"
        )

        print(
            f"Token Sayısı   : "
            f"{chunk.get('token_count', '-')}"
        )

        print("-" * 70)

        text = chunk.get(
            "text",
            "",
        ).strip()

        if len(text) > 700:
            text = text[:700] + " ..."

        print(text)

        print("-" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_arguments()

    # --------------------------------------------------
    # ARGUMENT KONTROLÜ
    # --------------------------------------------------

    if args.top_k <= 0:
        raise ValueError(
            "top_k değeri 0'dan büyük olmalıdır."
        )

    index_path, metadata_path = get_paths(
        args.strategy
    )

    print("=" * 70)
    print("HYBRID SEARCH")
    print("=" * 70)

    print(
        f"\nModel            : {MODEL_NAME}"
    )

    print(
        f"Chunk Stratejisi : "
        f"{args.strategy}"
    )

    print(
        f"Top-K            : "
        f"{args.top_k}"
    )

    # --------------------------------------------------
    # BGE-M3
    # --------------------------------------------------

    model, model_time = load_model()

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    faiss_index = load_index(
        index_path
    )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    metadata = load_metadata(
        metadata_path
    )

    # --------------------------------------------------
    # KONTROLLER
    # --------------------------------------------------

    if faiss_index.ntotal != len(metadata):

        raise ValueError(
            "FAISS index ve metadata "
            "sayıları eşleşmiyor!"
        )

    if faiss_index.d != 1024:

        raise ValueError(
            "FAISS vektör boyutu 1024 olmalıdır."
        )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25, bm25_time = build_bm25(
        metadata
    )

    print("\n" + "=" * 70)
    print("SİSTEM HAZIR")
    print("=" * 70)

    print(
        f"BGE-M3 yükleme : "
        f"{model_time:.2f} sn"
    )

    print(
        f"BM25 oluşturma  : "
        f"{bm25_time:.4f} sn"
    )

    # --------------------------------------------------
    # SORU DÖNGÜSÜ
    # --------------------------------------------------

    while True:

        print(
            "\nÇıkmak için 'q' yazabilirsiniz."
        )

        question = input(
            "\nSorunuzu girin:\n> "
        ).strip()

        if question.lower() == "q":

            print(
                "\nProgram sonlandırıldı."
            )

            break

        if not question:

            print(
                "\nBoş soru giremezsiniz."
            )

            continue

        # --------------------------------------------------
        # SEARCH
        # --------------------------------------------------

        start = time.perf_counter()

        try:

            results = hybrid_search(
                model=model,
                faiss_index=faiss_index,
                bm25=bm25,
                metadata=metadata,
                question=question,
                top_k=args.top_k,
            )

        except Exception as exc:

            print(
                "\nARAMA SIRASINDA HATA:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        elapsed = (
            time.perf_counter()
            - start
        )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        print_results(
            question,
            results,
        )

        print(
            f"\nToplam Hybrid Search süresi : "
            f"{elapsed:.4f} sn"
        )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()