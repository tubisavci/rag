"""
BM25 Semantic Retrieval

Kullanım:

python src/search_bm25.py --strategy 300_50

python src/search_bm25.py --strategy 500_100

python src/search_bm25.py --strategy 800_150
"""

import argparse
import json
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

TOP_K = 5


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():
    """Komut satırı argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description="BM25 Retrieval"
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
    """Metadata dosya yolunu oluşturur."""

    project_root = Path(__file__).resolve().parent.parent

    metadata_path = (
        project_root
        / "vector_db"
        / f"metadata_bge_m3_{strategy}.json"
    )

    return metadata_path


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
        f"Toplam chunk: {len(metadata)}"
    )

    return metadata


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    """
    Türkçe için basit regex tokenizer.

    Noktalama işaretlerini temizler.

    Küçük harfe dönüştürür.
    """

    text = text.lower()

    tokens = re.findall(
        r"\w+",
        text,
        flags=re.UNICODE,
    )

    return tokens


# --------------------------------------------------
# TOKENIZE TÜM CHUNKLAR
# --------------------------------------------------

def tokenize_documents(metadata):
    """Bütün chunkları tokenize eder."""

    print(
        "\nChunklar tokenize ediliyor..."
    )

    corpus = []

    for chunk in metadata:

        tokens = tokenize(
            chunk["text"]
        )

        corpus.append(tokens)

    print(
        f"Toplam tokenize edilen chunk: "
        f"{len(corpus)}"
    )

    return corpus


# --------------------------------------------------
# BM25 INDEX
# --------------------------------------------------

def build_bm25(corpus):
    """BM25 index oluşturur."""

    print(
        "\nBM25 index oluşturuluyor..."
    )

    start = time.perf_counter()

    bm25 = BM25Okapi(
        corpus
    )

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
# QUERY TOKENIZE
# --------------------------------------------------

def tokenize_query(question):
    """Sorguyu tokenize eder."""

    return tokenize(question)

# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def search(
    bm25,
    metadata,
    question,
    top_k,
):
    """BM25 ile arama yapar."""

    query_tokens = tokenize_query(
        question
    )

    start = time.perf_counter()

    scores = bm25.get_scores(
        query_tokens
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []

    for index, score in ranked:

        results.append(
            (
                index,
                score,
                metadata[index],
            )
        )

    return (
        results,
        elapsed,
    )


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

def print_results(
    question,
    results,
    elapsed,
):
    """BM25 sonuçlarını ekrana yazdırır."""

    print("\n" + "=" * 70)
    print("BM25 SEARCH SONUCU")
    print("=" * 70)

    print(f"\nSoru: {question}")

    print(
        f"Arama süresi: "
        f"{elapsed:.4f} saniye"
    )

    print(
        f"Toplam sonuç: "
        f"{len(results)}"
    )

    print("\n" + "=" * 70)

    for rank, (
        index,
        score,
        chunk,
    ) in enumerate(
        results,
        start=1,
    ):

        print(f"\n{rank}. SONUÇ")

        print(
            f"BM25 Skoru      : "
            f"{score:.4f}"
        )

        print(
            f"Chunk ID        : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Chunk Index     : "
            f"{chunk['chunk_index']}"
        )

        print(
            f"Token Sayısı    : "
            f"{chunk['token_count']}"
        )

        print(
            f"Kaynak          : "
            f"{chunk['source']}"
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

    metadata_path = get_paths(
        args.strategy
    )

    print("=" * 70)
    print("BM25 RETRIEVAL")
    print("=" * 70)

    print(
        f"\nChunk Stratejisi : {args.strategy}"
    )

    print(
        f"Top-K            : {args.top_k}"
    )

    print(
        f"Metadata         : {metadata_path}"
    )

    metadata = load_metadata(
        metadata_path
    )

    corpus = tokenize_documents(
        metadata
    )

    bm25, build_time = build_bm25(
        corpus
    )

    print(
        f"\nBM25 oluşturma süresi: "
        f"{build_time:.4f} sn"
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

        if question == "":

            print(
                "\nBoş soru giremezsiniz."
            )

            continue

        results, elapsed = search(
            bm25,
            metadata,
            question,
            args.top_k,
        )

        print_results(
            question,
            results,
            elapsed,
        )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
    