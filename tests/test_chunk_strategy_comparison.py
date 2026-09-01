"""
RAG - Chunk Strategy Comparison

37. Gün:
300/50, 500/100 ve 800/150 chunk stratejilerinin
Semantic Search performansını karşılaştırır.

Ölçülen metrikler:
- Top-1 Accuracy
- Recall@5
- MRR@5
- Ortalama arama süresi
- Chunk sayısı

Bağımlılıklar:
- BGE-M3
- FAISS
- NumPy

Pandas / sklearn kullanılmaz.
"""

import csv
import json
import sys
import time
from pathlib import Path


# =========================================================
# PROJE KÖK DİZİNİ
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# ÖNEMLİ:
# BGE / Torch önce import ediliyor.
# FAISS daha sonra import ediliyor.
#
# Windows'ta FAISS + Torch import sırası bazı
# ortamlarda native DLL / heap çakışmasına neden olabilir.
# =========================================================

from embedding_model import BGEEmbeddingModel

import numpy as np
import faiss


# =========================================================
# DİZİNLER
# =========================================================

CHUNKS_DIR = BASE_DIR / "data" / "chunks"

EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

RESULTS_DIR = BASE_DIR / "results"


# =========================================================
# CHUNK STRATEJİLERİ
# =========================================================

STRATEGIES = {
    "300_50": {
        "chunk_file": CHUNKS_DIR / "chunks_300_50.json",
        "embedding_file": (
            EMBEDDINGS_DIR / "embeddings_300_50.npy"
        ),
    },

    "500_100": {
        "chunk_file": CHUNKS_DIR / "chunks_500_100.json",
        "embedding_file": (
            EMBEDDINGS_DIR / "embeddings_500_100.npy"
        ),
    },

    "800_150": {
        "chunk_file": CHUNKS_DIR / "chunks_800_150.json",
        "embedding_file": (
            EMBEDDINGS_DIR / "embeddings_800_150.npy"
        ),
    },
}


# =========================================================
# BENCHMARK AYARLARI
# =========================================================

TOP_K = 5


# =========================================================
# TEST SORULARI
# =========================================================

QUESTIONS = [
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
        "question": (
            "Türkiye'nin uzay çalışmaları hangi kurum "
            "tarafından yürütülmektedir?"
        ),
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


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def load_chunks(path):
    """
    Chunk JSON dosyasını yükler.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)

    return chunks


def load_embeddings(path):
    """
    Önceden oluşturulmuş embedding dosyasını yükler.
    """

    embeddings = np.load(path)

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    return embeddings


def get_source_name(source):
    """
    Dosya adından benchmark'ta kullanılan kaynak adını çıkarır.

    Örnek:

    cevre_bakanlik_cevre-sorunlari-ve-oncelikleri.txt
    ->
    cevre_bakanlik
    """

    source = Path(source).stem

    known_sources = [
        "cevre_bakanlik",
        "dil_tdk",
        "egitim_meb",
        "enerji_etkb",
        "gida_tarimorman",
        "siber_guvenlik",
        "tarim_bakanlik",
        "tuketici_ticaretbakanligi",
        "uzay_tubitak",
        "yapayzeka",
    ]

    for known_source in known_sources:

        if source.startswith(known_source):

            return known_source

    return source


def build_faiss_index(embeddings):
    """
    Cosine similarity tabanlı Semantic Search için
    FAISS Inner Product index oluşturur.

    Embeddingler normalize edildiği için:

        Inner Product == Cosine Similarity
    """

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def calculate_top1(results, expected):
    """
    İlk sonucun beklenen kaynak olup olmadığını kontrol eder.
    """

    if not results:

        return 0

    return int(
        results[0]["source"] == expected
    )


def calculate_recall_at_k(results, expected):
    """
    Beklenen kaynak Top-K içerisinde bulunuyor mu?
    """

    for result in results:

        if result["source"] == expected:

            return 1

    return 0


def calculate_mrr_at_k(results, expected):
    """
    Mean Reciprocal Rank.

    Beklenen kaynak:

    1. sıradaysa -> 1.0
    2. sıradaysa -> 0.5
    3. sıradaysa -> 0.333
    ...

    Top-K dışında ise -> 0
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result["source"] == expected:

            return 1.0 / rank

    return 0.0


def search(index, chunks, query_embedding):
    """
    FAISS üzerinde Semantic Search yapar.
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    if query_embedding.ndim == 1:

        query_embedding = query_embedding.reshape(
            1,
            -1,
        )

    start = time.perf_counter()

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    elapsed = (
        time.perf_counter() - start
    )

    results = []

    for rank, chunk_index in enumerate(
        indices[0],
        start=1,
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        source = get_source_name(
            chunk["source"]
        )

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get(
                    "chunk_id",
                    chunk_index,
                ),
                "source": source,
                "score": float(
                    scores[0][rank - 1]
                ),
            }
        )

    return results, elapsed


# =========================================================
# ANA BENCHMARK
# =========================================================

def main():

    print("=" * 70)
    print("CHUNK STRATEGY COMPARISON")
    print("=" * 70)

    print(
        "\nStratejiler:"
    )

    for strategy in STRATEGIES:

        print(
            f"  - {strategy}"
        )

    print(
        f"\nTop-K: {TOP_K}"
    )

    print(
        f"Test sorusu sayısı: "
        f"{len(QUESTIONS)}"
    )

    # =====================================================
    # DOSYALARI KONTROL ET
    # =====================================================

    print("\n" + "=" * 70)
    print("DOSYA KONTROLÜ")
    print("=" * 70)

    for strategy, paths in STRATEGIES.items():

        chunk_file = paths["chunk_file"]

        embedding_file = paths[
            "embedding_file"
        ]

        if not chunk_file.exists():

            raise FileNotFoundError(
                f"Chunk dosyası bulunamadı:\n"
                f"{chunk_file}"
            )

        if not embedding_file.exists():

            raise FileNotFoundError(
                f"Embedding dosyası bulunamadı:\n"
                f"{embedding_file}"
            )

        print(
            f"\n{strategy}: OK"
        )

        print(
            f"  Chunk     : "
            f"{chunk_file.name}"
        )

        print(
            f"  Embedding : "
            f"{embedding_file.name}"
        )

    # =====================================================
    # MODEL
    # =====================================================

    print("\n" + "=" * 70)
    print("EMBEDDING MODELİ")
    print("=" * 70)

    model = BGEEmbeddingModel()

    # =====================================================
    # SORU EMBEDDINGLERİ
    #
    # Her strateji için tekrar embedding üretmek yerine
    # soruları sadece bir kez encode ediyoruz.
    # Böylece chunk stratejileri daha adil karşılaştırılır.
    # =====================================================

    questions_text = [
        item["question"]
        for item in QUESTIONS
    ]

    print(
        "\nTest soruları embedding'e dönüştürülüyor..."
    )

    embedding_start = time.perf_counter()

    question_embeddings = model.encode(
        questions_text
    )

    embedding_elapsed = (
        time.perf_counter()
        - embedding_start
    )

    print(
        "Soru embeddingleri hazır."
    )

    print(
        f"Shape: "
        f"{question_embeddings.shape}"
    )

    print(
        f"Süre: "
        f"{embedding_elapsed:.4f} sn"
    )

    # =====================================================
    # SONUÇLAR
    # =====================================================

    all_results = []

    # =====================================================
    # STRATEJİLER
    # =====================================================

    for strategy, paths in STRATEGIES.items():

        print("\n")
        print("=" * 70)

        print(
            f"STRATEJİ: {strategy}"
        )

        print("=" * 70)

        # -------------------------------------------------
        # CHUNKLARI YÜKLE
        # -------------------------------------------------

        print(
            "\nChunklar yükleniyor..."
        )

        chunks = load_chunks(
            paths["chunk_file"]
        )

        print(
            f"Toplam chunk: "
            f"{len(chunks)}"
        )

        # -------------------------------------------------
        # EMBEDDINGLERİ YÜKLE
        # -------------------------------------------------

        print(
            "\nEmbeddingler yükleniyor..."
        )

        embeddings = load_embeddings(
            paths["embedding_file"]
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        # -------------------------------------------------
        # KONTROL
        # -------------------------------------------------

        if len(chunks) != len(embeddings):

            raise ValueError(
                f"{strategy} için chunk ve "
                f"embedding sayıları eşleşmiyor!\n"
                f"Chunk: {len(chunks)}\n"
                f"Embedding: {len(embeddings)}"
            )

        # -------------------------------------------------
        # FAISS INDEX
        # -------------------------------------------------

        print(
            "\nFAISS index oluşturuluyor..."
        )

        index_start = time.perf_counter()

        index = build_faiss_index(
            embeddings
        )

        index_elapsed = (
            time.perf_counter()
            - index_start
        )

        print(
            f"FAISS hazır "
            f"({index_elapsed:.4f} sn)"
        )

        # -------------------------------------------------
        # SORULAR
        # -------------------------------------------------

        strategy_results = []

        top1_values = []

        recall_values = []

        mrr_values = []

        search_times = []

        for question_number, item in enumerate(
            QUESTIONS,
            start=1,
        ):

            question = item["question"]

            expected = item["expected"]

            print("\n" + "-" * 70)

            print(
                f"{question_number}/{len(QUESTIONS)}"
            )

            print(
                f"Soru: {question}"
            )

            print(
                f"Beklenen kaynak: {expected}"
            )

            # ---------------------------------------------
            # SEARCH
            # ---------------------------------------------

            results, search_time = search(
                index,
                chunks,
                question_embeddings[
                    question_number - 1
                ],
            )

            # ---------------------------------------------
            # METRİKLER
            # ---------------------------------------------

            top1 = calculate_top1(
                results,
                expected,
            )

            recall = calculate_recall_at_k(
                results,
                expected,
            )

            mrr = calculate_mrr_at_k(
                results,
                expected,
            )

            top1_values.append(top1)

            recall_values.append(recall)

            mrr_values.append(mrr)

            search_times.append(
                search_time
            )

            # ---------------------------------------------
            # SONUÇLAR
            # ---------------------------------------------

            print(
                f"\nTop-1: "
                f"{'EVET' if top1 else 'HAYIR'}"
            )

            print(
                f"Recall@{TOP_K}: "
                f"{'EVET' if recall else 'HAYIR'}"
            )

            print(
                f"MRR@{TOP_K}: "
                f"{mrr:.3f}"
            )

            print(
                f"Search süresi: "
                f"{search_time:.6f} sn"
            )

            print(
                "\nTop sonuçlar:"
            )

            for result in results:

                print(
                    f"  {result['rank']}. "
                    f"{result['source']} "
                    f"| score="
                    f"{result['score']:.4f}"
                )

            # ---------------------------------------------
            # CSV VERİSİ
            # ---------------------------------------------

            strategy_results.append(
                {
                    "strategy": strategy,
                    "question_number": question_number,
                    "question": question,
                    "expected_source": expected,
                    "top1": top1,
                    f"recall_at_{TOP_K}": recall,
                    f"mrr_at_{TOP_K}": round(
                        mrr,
                        6,
                    ),
                    "search_time_seconds": round(
                        search_time,
                        6,
                    ),
                    "top1_source": (
                        results[0]["source"]
                        if results
                        else ""
                    ),
                }
            )

        # -------------------------------------------------
        # STRATEJİ ÖZETİ
        # -------------------------------------------------

        top1_accuracy = (
            sum(top1_values)
            / len(top1_values)
        )

        recall_at_k = (
            sum(recall_values)
            / len(recall_values)
        )

        mrr_at_k = (
            sum(mrr_values)
            / len(mrr_values)
        )

        average_search_time = (
            sum(search_times)
            / len(search_times)
        )

        print("\n")
        print("=" * 70)

        print(
            f"{strategy} SONUÇLARI"
        )

        print("=" * 70)

        print(
            f"Chunk sayısı       : "
            f"{len(chunks)}"
        )

        print(
            f"Top-1 Accuracy     : "
            f"{top1_accuracy * 100:.2f}%"
        )

        print(
            f"Recall@{TOP_K}           : "
            f"{recall_at_k:.3f}"
        )

        print(
            f"MRR@{TOP_K}              : "
            f"{mrr_at_k:.3f}"
        )

        print(
            f"Ort. Search süresi : "
            f"{average_search_time:.6f} sn"
        )

        all_results.extend(
            strategy_results
        )

    # =====================================================
    # ÖZET HESAPLAMA
    # =====================================================

    summaries = []

    for strategy in STRATEGIES:

        strategy_rows = [
            row
            for row in all_results
            if row["strategy"] == strategy
        ]

        top1_accuracy = (
            sum(
                row["top1"]
                for row in strategy_rows
            )
            / len(strategy_rows)
        )

        recall_at_k = (
            sum(
                row[f"recall_at_{TOP_K}"]
                for row in strategy_rows
            )
            / len(strategy_rows)
        )

        mrr_at_k = (
            sum(
                row[f"mrr_at_{TOP_K}"]
                for row in strategy_rows
            )
            / len(strategy_rows)
        )

        average_time = (
            sum(
                row["search_time_seconds"]
                for row in strategy_rows
            )
            / len(strategy_rows)
        )

        chunk_count = sum(
            1
            for _ in load_chunks(
                STRATEGIES[strategy][
                    "chunk_file"
                ]
            )
        )

        summaries.append(
            {
                "strategy": strategy,
                "chunk_count": chunk_count,
                "top1_accuracy": round(
                    top1_accuracy,
                    6,
                ),
                f"recall_at_{TOP_K}": round(
                    recall_at_k,
                    6,
                ),
                f"mrr_at_{TOP_K}": round(
                    mrr_at_k,
                    6,
                ),
                "average_search_time_seconds": round(
                    average_time,
                    6,
                ),
            }
        )

    # =====================================================
    # SONUÇLARI KONSOLA YAZ
    # =====================================================

    print("\n")
    print("=" * 70)

    print("CHUNK STRATEJİLERİ KARŞILAŞTIRMA SONUÇLARI")

    print("=" * 70)

    print()

    print(
        f"{'Strateji':<12}"
        f"{'Chunk':<10}"
        f"{'Top-1':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR@5':<12}"
        f"{'Süre (sn)':<12}"
    )

    print("-" * 70)

    for summary in summaries:

        print(
            f"{summary['strategy']:<12}"
            f"{summary['chunk_count']:<10}"
            f"{summary['top1_accuracy'] * 100:>7.2f}%   "
            f"{summary[f'recall_at_{TOP_K}']:>8.3f}    "
            f"{summary[f'mrr_at_{TOP_K}']:>8.3f}    "
            f"{summary['average_search_time_seconds']:>10.6f}"
        )

    # =====================================================
    # CSV KAYDET
    # =====================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_csv = (
        RESULTS_DIR
        / "chunk_strategy_comparison_details.csv"
    )

    summary_csv = (
        RESULTS_DIR
        / "chunk_strategy_comparison_summary.csv"
    )

    # -----------------------------------------------------
    # DETAY CSV
    # -----------------------------------------------------

    detail_fields = [
        "strategy",
        "question_number",
        "question",
        "expected_source",
        "top1",
        f"recall_at_{TOP_K}",
        f"mrr_at_{TOP_K}",
        "search_time_seconds",
        "top1_source",
    ]

    with detail_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # -----------------------------------------------------
    # ÖZET CSV
    # -----------------------------------------------------

    summary_fields = [
        "strategy",
        "chunk_count",
        "top1_accuracy",
        f"recall_at_{TOP_K}",
        f"mrr_at_{TOP_K}",
        "average_search_time_seconds",
    ]

    with summary_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            summaries
        )

    # =====================================================
    # TAMAMLANDI
    # =====================================================

    print("\n" + "=" * 70)

    print("BENCHMARK TAMAMLANDI")

    print("=" * 70)

    print(
        f"\nDetay CSV:"
        f"\n{detail_csv}"
    )

    print(
        f"\nÖzet CSV:"
        f"\n{summary_csv}"
    )

    print(
        "\n37. gün chunk stratejisi "
        "karşılaştırması tamamlandı."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()