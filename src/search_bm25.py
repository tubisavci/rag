"""
BM25 Retrieval

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
# SABİTLER
# --------------------------------------------------

TOP_K = 5


# --------------------------------------------------
# ARGÜMANLAR
# --------------------------------------------------

def parse_arguments():
    """Komut satırı argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description="BM25 Retrieval"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
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

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error(
            "top_k değeri 0'dan büyük olmalıdır."
        )

    return args


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

def get_paths(strategy):
    """Chunk dosya yolunu oluşturur."""

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    chunks_path = (
        project_root
        / "data"
        / "chunks"
        / f"chunks_{strategy}.json"
    )

    return chunks_path


# --------------------------------------------------
# CHUNKLARI YÜKLE
# --------------------------------------------------

def load_chunks(chunks_path):
    """Chunk JSON dosyasını yükler."""

    print("\nChunklar yükleniyor...")

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk dosyası bulunamadı: "
            f"{chunks_path}"
        )

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "Chunk dosyası liste formatında "
            "olmalıdır."
        )

    if not chunks:
        raise ValueError(
            "Chunk dosyası boş olamaz."
        )

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"{index}. chunk dictionary "
                f"formatında olmalıdır."
            )

        if "text" not in chunk:
            raise ValueError(
                f"{index}. chunk içerisinde "
                f"'text' alanı bulunamadı."
            )

        if not isinstance(
            chunk["text"],
            str,
        ):
            raise ValueError(
                f"{index}. chunk içerisindeki "
                f"'text' alanı string olmalıdır."
            )

    print(
        f"Toplam chunk: {len(chunks)}"
    )

    return chunks


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    """
    Basit Türkçe regex tokenizer.

    Metni küçük harfe dönüştürür ve
    kelime tabanlı tokenlar üretir.
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    return re.findall(
        r"\w+",
        text,
        flags=re.UNICODE,
    )


# --------------------------------------------------
# TÜM CHUNKLARI TOKENIZE ET
# --------------------------------------------------

def tokenize_documents(chunks):
    """Bütün chunkları tokenize eder."""

    print(
        "\nChunklar tokenize ediliyor..."
    )

    corpus = []

    for chunk in chunks:

        tokens = tokenize(
            chunk["text"]
        )

        corpus.append(tokens)

    if not corpus:
        raise ValueError(
            "Tokenize edilen corpus boş."
        )

    empty_documents = sum(
        1
        for tokens in corpus
        if not tokens
    )

    if empty_documents > 0:
        print(
            f"Uyarı: {empty_documents} "
            f"chunk boş token içeriyor."
        )

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

    if not corpus:
        raise ValueError(
            "BM25 corpus boş olamaz."
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
        f"BM25 index hazır "
        f"({elapsed:.4f} saniye)"
    )

    return bm25, elapsed


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
    chunks,
    question,
    top_k,
):
    """BM25 ile arama yapar."""

    if top_k <= 0:
        raise ValueError(
            "top_k değeri 0'dan büyük "
            "olmalıdır."
        )

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question string olmalıdır."
        )

    if not question.strip():
        return [], 0.0

    query_tokens = tokenize_query(
        question
    )

    if not query_tokens:
        return [], 0.0

    start = time.perf_counter()

    scores = bm25.get_scores(
        query_tokens
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
                int(index),
                float(score),
                chunks[index],
            )
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return results, elapsed


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

def print_results(
    question,
    results,
    elapsed,
):
    """BM25 sonuçlarını ekrana yazdırır."""

    print(
        "\n" + "=" * 70
    )

    print(
        "BM25 SEARCH SONUCU"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSoru: {question}"
    )

    print(
        f"Arama süresi: "
        f"{elapsed:.4f} saniye"
    )

    print(
        f"Toplam sonuç: "
        f"{len(results)}"
    )

    print(
        "\n" + "=" * 70
    )

    for rank, (
        index,
        score,
        chunk,
    ) in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n{rank}. SONUÇ"
        )

        print(
            f"BM25 Skoru      : "
            f"{score:.4f}"
        )

        print(
            f"Chunk ID        : "
            f"{chunk.get('chunk_id', '-')}"
        )

        print(
            f"Chunk Index     : "
            f"{chunk.get('chunk_index', index)}"
        )

        print(
            f"Token Sayısı    : "
            f"{chunk.get('token_count', '-')}"
        )

        print(
            f"Kaynak          : "
            f"{chunk.get('source', '-')}"
        )

        print(
            "-" * 70
        )

        text = chunk.get(
            "text",
            "",
        ).strip()

        if len(text) > 700:
            text = (
                text[:700]
                + " ..."
            )

        print(text)

        print(
            "-" * 70
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_arguments()

    chunks_path = get_paths(
        args.strategy
    )

    print(
        "=" * 70
    )

    print(
        "BM25 RETRIEVAL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nChunk Stratejisi : "
        f"{args.strategy}"
    )

    print(
        f"Top-K            : "
        f"{args.top_k}"
    )

    print(
        f"Chunks           : "
        f"{chunks_path}"
    )

    # --------------------------------------------------
    # CHUNKLARI YÜKLE
    # --------------------------------------------------

    chunks = load_chunks(
        chunks_path
    )

    # --------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------

    corpus = tokenize_documents(
        chunks
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25, build_time = build_bm25(
        corpus
    )

    print(
        f"\nBM25 oluşturma süresi: "
        f"{build_time:.4f} saniye"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SİSTEM HAZIR"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------
    # SORU DÖNGÜSÜ
    # --------------------------------------------------

    while True:

        print(
            "\nÇıkmak için 'q' "
            "yazabilirsiniz."
        )

        question = input(
            "\nSorunuzu girin: "
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

        try:

            results, elapsed = search(
                bm25,
                chunks,
                question,
                args.top_k,
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