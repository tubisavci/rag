"""
Hybrid Search (Semantic + BM25)

Kullanım:

python src/search_hybrid.py --strategy 300_50

python src/search_hybrid.py --strategy 500_100

python src/search_hybrid.py --strategy 800_150
"""

import argparse
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

TOP_K = 5

SEMANTIC_WEIGHT = 0.6

BM25_WEIGHT = 0.4

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
    """Dosya yollarını oluşturur."""

    project_root = Path(__file__).resolve().parent.parent

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

    return (
        index_path,
        metadata_path,
    )

# --------------------------------------------------
# FAISS INDEX
# --------------------------------------------------

def load_index(index_path):
    """FAISS indexini yükler."""

    print("\nFAISS index yükleniyor...")

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    with index_path.open("rb") as file:

        serialized = np.frombuffer(
            file.read(),
            dtype="uint8",
        )

    index = faiss.deserialize_index(
        serialized
    )

    print(
        f"Toplam vektör : "
        f"{index.ntotal}"
    )

    return index


# --------------------------------------------------
# METADATA
# --------------------------------------------------

def load_metadata(metadata_path):
    """Metadata dosyasını yükler."""

    print("\nMetadata yükleniyor...")

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    print(
        f"Toplam metadata : "
        f"{len(metadata)}"
    )

    return metadata


# --------------------------------------------------
# EMBEDDING MODELİ
# --------------------------------------------------

def load_model():
    """BGE-M3 modelini yükler."""

    print("\nEmbedding modeli yükleniyor...")

    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

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
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    """Türkçe tokenizer."""

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


# --------------------------------------------------
# BM25
# --------------------------------------------------

def build_bm25(metadata):
    """BM25 index oluşturur."""

    print("\nBM25 index oluşturuluyor...")

    corpus = [
        tokenize(chunk["text"])
        for chunk in metadata
    ]

    start = time.perf_counter()

    bm25 = BM25Okapi(corpus)

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"BM25 hazır "
        f"({elapsed:.4f} sn)"
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
    FAISS üzerinde semantic search yapar.
    """

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0],
    ):
        results.append(
            (
                int(idx),
                float(score),
            )
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
    """
    BM25 retrieval yapar.
    """

    query_tokens = tokenize(question)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []

    for idx, score in ranked:

        results.append(
            (
                int(idx),
                float(score),
            )
        )

    return results

# --------------------------------------------------
# MIN-MAX NORMALIZATION
# --------------------------------------------------

def normalize_scores(results):
    """
    Skorları 0-1 aralığına normalize eder.
    """

    if not results:
        return {}

    scores = [
        score
        for _, score in results
    ]

    minimum = min(scores)
    maximum = max(scores)

    normalized = {}

    if maximum == minimum:

        for idx, _ in results:
            normalized[idx] = 1.0

        return normalized

    for idx, score in results:

        normalized[idx] = (
            (score - minimum)
            /
            (maximum - minimum)
        )

    return normalized

# --------------------------------------------------
# HYBRID SEARCH
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
    Semantic + BM25 skorlarını birleştirir.
    """

    semantic_results = semantic_search(
        model,
        faiss_index,
        question,
        top_k * 3,
    )

    bm25_results = bm25_search(
        bm25,
        question,
        top_k * 3,
    )

    semantic_scores = normalize_scores(
        semantic_results
    )

    bm25_scores = normalize_scores(
        bm25_results
    )

    all_indices = set()

    all_indices.update(
        semantic_scores.keys()
    )

    all_indices.update(
        bm25_scores.keys()
    )

    hybrid_results = []

    for idx in all_indices:

        semantic = semantic_scores.get(
            idx,
            0,
        )

        lexical = bm25_scores.get(
            idx,
            0,
        )

        score = (
            SEMANTIC_WEIGHT
            * semantic
            +
            BM25_WEIGHT
            * lexical
        )

        hybrid_results.append(
            (
                idx,
                score,
            )
        )

    hybrid_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    final_results = []

    for idx, score in hybrid_results[:top_k]:

        final_results.append(
            (
                metadata[idx],
                score,
            )
        )

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

    print(f"\nSoru: {question}")

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

        print(f"\n{rank}. SONUÇ")

        print(
            f"Hybrid Skor    : {score:.4f}"
        )

        print(
            f"Kaynak         : {chunk['source']}"
        )

        print(
            f"Chunk ID       : {chunk['chunk_id']}"
        )

        print(
            f"Chunk Index    : {chunk['chunk_index']}"
        )

        print(
            f"Token Sayısı   : {chunk['token_count']}"
        )

        print("-" * 70)

        text = chunk["text"].strip()

        if len(text) > 700:
            text = text[:700] + " ..."

        print(text)

        print("-" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_arguments()

    index_path, metadata_path = get_paths(
        args.strategy
    )

    print("=" * 70)
    print("HYBRID SEARCH")
    print("=" * 70)

    print(f"\nModel            : {MODEL_NAME}")
    print(f"Chunk Stratejisi : {args.strategy}")
    print(f"Top-K            : {args.top_k}")

    faiss_index = load_index(
        index_path
    )

    metadata = load_metadata(
        metadata_path
    )

    model, _ = load_model()

    bm25, _ = build_bm25(
        metadata
    )

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

        start = time.perf_counter()

        results = hybrid_search(
            model=model,
            faiss_index=faiss_index,
            bm25=bm25,
            metadata=metadata,
            question=question,
            top_k=args.top_k,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print_results(
            question,
            results,
        )

        print(
            f"\nHybrid arama süresi : "
            f"{elapsed:.4f} sn"
        )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()

    