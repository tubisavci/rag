"""
Chunk-Level Relevance Candidate Generator

Mevcut retrieval benchmark sonuçlarından:
- Semantic Search
- BM25
- Hybrid Search
- Hybrid + Reranker

tarafından getirilen Top-5 chunk'ları toplar.

Bu script ground truth oluşturmaz.
Sadece insan tarafından relevance değerlendirmesi yapılacak
aday chunk'ları hazırlar.
"""

import csv
import json
from pathlib import Path


# ==========================================================
# AYARLAR
# ==========================================================

CHUNK_STRATEGY = "500_100"

TOP_K = 5


# ==========================================================
# PATHLER
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "benchmark_results.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "vector_db"
    / f"metadata_bge_m3_{CHUNK_STRATEGY}.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "chunk_relevance_candidates.json"
)


# ==========================================================
# KONTROLLER
# ==========================================================

if not BENCHMARK_PATH.exists():
    raise FileNotFoundError(
        f"Benchmark dosyası bulunamadı: {BENCHMARK_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata dosyası bulunamadı: {METADATA_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# BENCHMARK OKUMA
# ==========================================================

print("=" * 70)
print("CHUNK-LEVEL RELEVANCE ADAYLARI")
print("=" * 70)

print(
    f"\nChunk stratejisi : {CHUNK_STRATEGY}"
)

print(
    f"Top-K            : {TOP_K}"
)

print(
    f"\nBenchmark okunuyor..."
)

with BENCHMARK_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:

    reader = csv.DictReader(file)

    benchmark_rows = list(reader)


if not benchmark_rows:
    raise ValueError(
        "Benchmark dosyası boş."
    )


print(
    f"Toplam soru : {len(benchmark_rows)}"
)


# ==========================================================
# METADATA OKUMA
# ==========================================================

print(
    "\nMetadata okunuyor..."
)

with METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


if not isinstance(metadata, list):
    raise ValueError(
        "Metadata list formatında olmalıdır."
    )


print(
    f"Toplam chunk : {len(metadata)}"
)


# ==========================================================
# CHUNK ID PARSE
# ==========================================================

def parse_chunk_ids(value):
    """
    CSV içerisindeki:

        12|25|48|51|72

    formatındaki chunk ID'lerini listeye çevirir.
    """

    if not value:
        return []

    ids = []

    for item in value.split("|"):

        item = item.strip()

        if not item:
            continue

        try:
            ids.append(
                int(item)
            )

        except ValueError:
            print(
                f"UYARI: Geçersiz chunk ID: {item}"
            )

    return ids


# ==========================================================
# SONUÇLARI OLUŞTUR
# ==========================================================

all_questions = []


for question_number, row in enumerate(
    benchmark_rows,
    start=1,
):

    question = row["question"]

    expected_source = row["expected"]

    semantic_ids = parse_chunk_ids(
        row["semantic_top5"]
    )

    bm25_ids = parse_chunk_ids(
        row["bm25_top5"]
    )

    hybrid_ids = parse_chunk_ids(
        row["hybrid_top5"]
    )

    reranker_ids = parse_chunk_ids(
        row["reranker_top5"]
    )


    # ------------------------------------------------------
    # TÜM ADAYLARI BİRLEŞTİR
    # ------------------------------------------------------

    candidate_ids = []

    for chunk_id in (
        semantic_ids
        + bm25_ids
        + hybrid_ids
        + reranker_ids
    ):

        if chunk_id not in candidate_ids:
            candidate_ids.append(
                chunk_id
            )


    # ------------------------------------------------------
    # CHUNK BİLGİLERİ
    # ------------------------------------------------------

    candidates = []

    for chunk_id in candidate_ids:

        if chunk_id < 0:
            continue

        if chunk_id >= len(metadata):

            print(
                f"UYARI: Chunk ID metadata dışında: "
                f"{chunk_id}"
            )

            continue


        chunk = metadata[chunk_id]


        if not isinstance(chunk, dict):

            print(
                f"UYARI: Metadata[{chunk_id}] "
                f"dict değil."
            )

            continue


        candidates.append({

            "chunk_id": chunk_id,

            "source": chunk.get(
                "source",
                "",
            ),

            "chunk_index": chunk.get(
                "chunk_index",
                "",
            ),

            "token_count": chunk.get(
                "token_count",
                "",
            ),

            "text": chunk.get(
                "text",
                "",
            ),

            # --------------------------------------------------
            # Hangi retrieval yöntemlerinde bulundu?
            # --------------------------------------------------

            "found_by": {

                "semantic": (
                    chunk_id in semantic_ids
                ),

                "bm25": (
                    chunk_id in bm25_ids
                ),

                "hybrid": (
                    chunk_id in hybrid_ids
                ),

                "reranker": (
                    chunk_id in reranker_ids
                ),

            },

            # --------------------------------------------------
            # Ground truth henüz belirlenmedi.
            # --------------------------------------------------

            "relevant": None,

        })


    # ------------------------------------------------------
    # SORU KAYDI
    # ------------------------------------------------------

    all_questions.append({

        "question_id": question_number,

        "question": question,

        "expected_source": expected_source,

        "candidates": candidates,

    })


# ==========================================================
# JSON KAYDET
# ==========================================================

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        all_questions,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ==========================================================
# ÖZET
# ==========================================================

total_candidates = sum(
    len(item["candidates"])
    for item in all_questions
)

print(
    "\n" + "=" * 70
)

print(
    "ADAYLAR OLUŞTURULDU"
)

print(
    "=" * 70
)

print(
    f"Soru sayısı              : "
    f"{len(all_questions)}"
)

print(
    f"Toplam benzersiz aday    : "
    f"{total_candidates}"
)

print(
    f"\nÇıktı dosyası:"
)

print(
    OUTPUT_PATH
)

print(
    "\nNOT:"
)

print(
    "Bu dosyadaki 'relevant' alanları "
    "henüz ground truth değildir."
)

print(
    "İlgililik değerlendirmesi "
    "doküman metinleri incelenerek yapılacaktır."
)