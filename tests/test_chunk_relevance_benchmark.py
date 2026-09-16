"""
Chunk-Level Relevance Benchmark

Manuel olarak etiketlenen ground truth kullanılarak
retrieval sonuçlarının chunk-level performansını ölçer.

Relevance:
0 = İlgisiz
1 = Kısmen ilgili
2 = Doğrudan ilgili

Metrikler:
- Precision@K
- Recall@K
- MRR@K
- nDCG@K
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import math


# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_PATH = (
    BASE_DIR / "results" / "chunk_relevance_ground_truth.json"
)

BENCHMARK_PATH = BASE_DIR / "benchmark_results.csv"

OUTPUT_PATH = (
    BASE_DIR / "results" / "chunk_level_benchmark_results.csv"
)

TOP_K = 5


# ============================================================
# DOSYA OKUMA
# ============================================================

def load_ground_truth():
    """Manuel relevance etiketlerini yükler."""

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth bulunamadı:\n{GROUND_TRUTH_PATH}"
        )

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_benchmark_results():
    """Mevcut retrieval benchmark sonuçlarını yükler."""

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark dosyası bulunamadı:\n{BENCHMARK_PATH}"
        )

    with open(BENCHMARK_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# GROUND TRUTH HAZIRLAMA
# ============================================================

def build_relevance_map(ground_truth):
    """
    Her soru için:
        chunk_id -> relevance

    haritası oluşturur.
    """

    relevance_map = defaultdict(dict)

    for question_data in ground_truth:

        question = question_data["question"]

        candidates = question_data.get("candidates", [])

        for candidate in candidates:

            chunk_id = str(candidate["chunk_id"])
            relevance = candidate.get("relevant")

            if relevance is None:
                continue

            relevance_map[question][chunk_id] = int(relevance)

    return relevance_map


# ============================================================
# RETRIEVAL SONUÇLARINI OKUMA
# ============================================================

def parse_chunk_ids(value):
    """
    benchmark_results.csv içerisindeki Top-5 chunk ID listesini okur.

    Format:
        14|16|7|6|11
    """

    if not value:
        return []

    value = value.strip()

    # Benchmark dosyasındaki gerçek format: |
    if "|" in value:
        return [
            x.strip()
            for x in value.split("|")
            if x.strip()
        ]

    # JSON formatını da destekle
    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return [str(x) for x in parsed]

    except json.JSONDecodeError:
        pass

    # Virgülle ayrılmış formatı destekle
    value = value.strip("[]")

    if not value:
        return []

    return [
        x.strip().strip("'").strip('"')
        for x in value.split(",")
        if x.strip()
    ]

# ============================================================
# METRİKLER
# ============================================================

def precision_at_k(relevances, k):
    """
    Precision@K

    relevance > 0 olan chunk'lar ilgili kabul edilir.
    """

    top_k = relevances[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(1 for r in top_k if r > 0)

    return relevant_count / len(top_k)


def recall_at_k(relevances, k, total_relevant):
    """
    Recall@K

    relevance > 0 olan chunk'lar ilgili kabul edilir.
    """

    if total_relevant == 0:
        return 0.0

    top_k = relevances[:k]

    retrieved_relevant = sum(
        1 for r in top_k if r > 0
    )

    return retrieved_relevant / total_relevant


def mrr_at_k(relevances, k):
    """
    MRR@K

    İlk ilgili chunk'ın sırasına bakar.
    """

    for rank, relevance in enumerate(relevances[:k], start=1):

        if relevance > 0:
            return 1.0 / rank

    return 0.0


def dcg_at_k(relevances, k):
    """
    DCG@K

    Relevance değerleri:
        0 = 0
        1 = kısmen ilgili
        2 = doğrudan ilgili
    """

    score = 0.0

    for rank, relevance in enumerate(
        relevances[:k],
        start=1
    ):

        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(relevances, k):
    """
    nDCG@K
    """

    actual = dcg_at_k(relevances, k)

    ideal = sorted(
        relevances,
        reverse=True
    )

    ideal = dcg_at_k(ideal, k)

    if ideal == 0:
        return 0.0

    return actual / ideal


# ============================================================
# METOD ÇÖZÜMLEME
# ============================================================

def get_methods(row):
    """
    benchmark_results.csv sütunlarından
    retrieval yöntemlerini belirler.

    Her yöntem için Top-5 chunk ID'leri okunur.
    """

    methods = {}

    column_mapping = {
        "semantic": "semantic_top5",
        "bm25": "bm25_top5",
        "hybrid": "hybrid_top5",
        "reranker": "reranker_top5",
    }

    for method, column in column_mapping.items():

        if column in row:

            methods[method] = parse_chunk_ids(
                row[column]
            )

    return methods


# ============================================================
# ANA BENCHMARK
# ============================================================

def main():

    print("=" * 80)
    print("CHUNK-LEVEL RELEVANCE BENCHMARK")
    print("=" * 80)

    print()

    print("Ground truth:")
    print(GROUND_TRUTH_PATH)

    print()

    ground_truth = load_ground_truth()
    benchmark_rows = load_benchmark_results()

    relevance_map = build_relevance_map(
        ground_truth
    )

    print(
        f"Ground truth soru sayısı: "
        f"{len(relevance_map)}"
    )

    print(
        f"Benchmark soru sayısı: "
        f"{len(benchmark_rows)}"
    )

    print()

    results = []

    method_metrics = defaultdict(
        lambda: {
            "precision": [],
            "recall": [],
            "mrr": [],
            "ndcg": [],
        }
    )

    # ========================================================
    # HER SORUYU DEĞERLENDİR
    # ========================================================

    for question_number, row in enumerate(
        benchmark_rows,
        start=1
    ):

        question = row["question"]

        print("-" * 80)
        print(f"SORU {question_number}")
        print(question)
        print()

        question_relevance = relevance_map.get(
            question,
            {}
        )

        # Toplam relevant chunk
        total_relevant = sum(
            1
            for relevance in question_relevance.values()
            if relevance > 0
        )

        methods = get_methods(row)

        for method, chunk_ids in methods.items():

            # Sadece ilk TOP_K
            chunk_ids = chunk_ids[:TOP_K]

            relevances = [
                question_relevance.get(
                    chunk_id,
                    0
                )
                for chunk_id in chunk_ids
            ]

            precision = precision_at_k(
                relevances,
                TOP_K
            )

            recall = recall_at_k(
                relevances,
                TOP_K,
                total_relevant
            )

            mrr = mrr_at_k(
                relevances,
                TOP_K
            )

            ndcg = ndcg_at_k(
                relevances,
                TOP_K
            )

            method_metrics[method][
                "precision"
            ].append(precision)

            method_metrics[method][
                "recall"
            ].append(recall)

            method_metrics[method][
                "mrr"
            ].append(mrr)

            method_metrics[method][
                "ndcg"
            ].append(ndcg)

            print(
                f"{method.upper():10s} | "
                f"P@{TOP_K}: {precision:.3f} | "
                f"R@{TOP_K}: {recall:.3f} | "
                f"MRR@{TOP_K}: {mrr:.3f} | "
                f"nDCG@{TOP_K}: {ndcg:.3f}"
            )

            results.append({
                "question": question,
                "method": method,
                "precision_at_5": precision,
                "recall_at_5": recall,
                "mrr_at_5": mrr,
                "ndcg_at_5": ndcg,
            })

        print()

    # ========================================================
    # ORTALAMA SONUÇLAR
    # ========================================================

    print()
    print("=" * 80)
    print("ORTALAMA CHUNK-LEVEL SONUÇLAR")
    print("=" * 80)

    summary_rows = []

    for method, metrics in method_metrics.items():

        avg_precision = (
            sum(metrics["precision"])
            / len(metrics["precision"])
        )

        avg_recall = (
            sum(metrics["recall"])
            / len(metrics["recall"])
        )

        avg_mrr = (
            sum(metrics["mrr"])
            / len(metrics["mrr"])
        )

        avg_ndcg = (
            sum(metrics["ndcg"])
            / len(metrics["ndcg"])
        )

        print(
            f"{method.upper():10s} | "
            f"P@5: {avg_precision:.3f} | "
            f"R@5: {avg_recall:.3f} | "
            f"MRR@5: {avg_mrr:.3f} | "
            f"nDCG@5: {avg_ndcg:.3f}"
        )

        summary_rows.append({
            "method": method,
            "precision_at_5": avg_precision,
            "recall_at_5": avg_recall,
            "mrr_at_5": avg_mrr,
            "ndcg_at_5": avg_ndcg,
        })

    # ========================================================
    # SONUÇLARI KAYDET
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "question",
            "method",
            "precision_at_5",
            "recall_at_5",
            "mrr_at_5",
            "ndcg_at_5",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    summary_path = (
        BASE_DIR
        / "results"
        / "chunk_level_benchmark_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "method",
            "precision_at_5",
            "recall_at_5",
            "mrr_at_5",
            "ndcg_at_5",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(summary_rows)

    print()
    print("=" * 80)
    print("BENCHMARK TAMAMLANDI")
    print("=" * 80)

    print()
    print("Detay:")
    print(OUTPUT_PATH)

    print()
    print("Özet:")
    print(summary_path)


if __name__ == "__main__":
    main()