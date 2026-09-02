import csv
from pathlib import Path


# --------------------------------------------------
# DOSYA YOLU
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "benchmark_results.csv"


# --------------------------------------------------
# CSV OKU
# --------------------------------------------------

if not CSV_PATH.exists():
    print("benchmark_results.csv bulunamadı.")
    raise SystemExit


with CSV_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as file:

    reader = csv.DictReader(file)
    results = list(reader)


if not results:
    print("Benchmark sonucu bulunamadı.")
    raise SystemExit


# --------------------------------------------------
# BAŞLIK
# --------------------------------------------------

print("=" * 70)
print("RAG RETRIEVAL PERFORMANS ANALİZİ")
print("=" * 70)

print(
    f"\nToplam test sorusu : {len(results)}"
)


# --------------------------------------------------
# YARDIMCI FONKSİYON
# --------------------------------------------------

def average(values):
    """Liste ortalamasını hesaplar."""

    if not values:
        return 0.0

    return sum(values) / len(values)


# --------------------------------------------------
# SÜRELER
# --------------------------------------------------

semantic_times = [
    float(row["semantic_time"])
    for row in results
]

bm25_times = [
    float(row["bm25_time"])
    for row in results
]

hybrid_times = [
    float(row["hybrid_time"])
    for row in results
]

reranker_times = [
    float(row["reranker_time"])
    for row in results
]


semantic_avg = average(semantic_times)
bm25_avg = average(bm25_times)
hybrid_avg = average(hybrid_times)
reranker_avg = average(reranker_times)


# --------------------------------------------------
# DOĞRULUK
# --------------------------------------------------

semantic_correct = sum(
    1
    for row in results
    if row["semantic_correct"].lower() == "true"
)

bm25_correct = sum(
    1
    for row in results
    if row["bm25_correct"].lower() == "true"
)

hybrid_correct = sum(
    1
    for row in results
    if row["hybrid_correct"].lower() == "true"
)

reranker_correct = sum(
    1
    for row in results
    if row["reranker_correct"].lower() == "true"
)


total_questions = len(results)


semantic_accuracy = (
    semantic_correct / total_questions * 100
)

bm25_accuracy = (
    bm25_correct / total_questions * 100
)

hybrid_accuracy = (
    hybrid_correct / total_questions * 100
)

reranker_accuracy = (
    reranker_correct / total_questions * 100
)


# --------------------------------------------------
# RERANKER DEĞİŞİM ANALİZİ
# --------------------------------------------------

reranker_changed = sum(
    1
    for row in results
    if row["reranker_changed"].lower() == "true"
)

reranker_change_rate = (
    reranker_changed / total_questions * 100
)


# --------------------------------------------------
# SONUÇLAR
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. DOĞRULUK ANALİZİ")
print("=" * 70)

print(
    f"\nSemantic Search       : "
    f"{semantic_correct}/{total_questions} "
    f"({semantic_accuracy:.2f}%)"
)

print(
    f"BM25                  : "
    f"{bm25_correct}/{total_questions} "
    f"({bm25_accuracy:.2f}%)"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_correct}/{total_questions} "
    f"({hybrid_accuracy:.2f}%)"
)

print(
    f"Hybrid + Reranker     : "
    f"{reranker_correct}/{total_questions} "
    f"({reranker_accuracy:.2f}%)"
)


# --------------------------------------------------
# SÜRE ANALİZİ
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. ORTALAMA ARAMA SÜRELERİ")
print("=" * 70)

print(
    f"\nSemantic Search       : "
    f"{semantic_avg:.4f} sn"
)

print(
    f"BM25                  : "
    f"{bm25_avg:.4f} sn"
)

print(
    f"Hybrid + RRF          : "
    f"{hybrid_avg:.4f} sn"
)

print(
    f"Reranker              : "
    f"{reranker_avg:.4f} sn"
)


# --------------------------------------------------
# TOPLAM SÜRE
# --------------------------------------------------

total_hybrid_reranker_times = [
    float(row["hybrid_time"])
    + float(row["reranker_time"])
    for row in results
]

hybrid_reranker_avg = average(
    total_hybrid_reranker_times
)


print(
    f"Hybrid + Reranker Toplam : "
    f"{hybrid_reranker_avg:.4f} sn"
)


# --------------------------------------------------
# RERANKER ANALİZİ
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. RERANKER ANALİZİ")
print("=" * 70)

print(
    f"\nSıralaması değişen sonuç : "
    f"{reranker_changed}/{total_questions}"
)

print(
    f"Değişim oranı            : "
    f"{reranker_change_rate:.2f}%"
)


# --------------------------------------------------
# KARŞILAŞTIRMA
# --------------------------------------------------

print("\n" + "=" * 70)
print("4. YÖNTEM KARŞILAŞTIRMASI")
print("=" * 70)

print()

print(
    f"{'Yöntem':<25}"
    f"{'Doğruluk':>12}"
    f"{'Ort. Süre':>15}"
)

print("-" * 52)

print(
    f"{'Semantic Search':<25}"
    f"{semantic_accuracy:>10.2f}%"
    f"{semantic_avg:>13.4f} sn"
)

print(
    f"{'BM25':<25}"
    f"{bm25_accuracy:>10.2f}%"
    f"{bm25_avg:>13.4f} sn"
)

print(
    f"{'Hybrid + RRF':<25}"
    f"{hybrid_accuracy:>10.2f}%"
    f"{hybrid_avg:>13.4f} sn"
)

print(
    f"{'Hybrid + Reranker':<25}"
    f"{reranker_accuracy:>10.2f}%"
    f"{hybrid_reranker_avg:>13.4f} sn"
)


# --------------------------------------------------
# GELİŞİM
# --------------------------------------------------

print("\n" + "=" * 70)
print("5. GELİŞİM")
print("=" * 70)

hybrid_improvement = (
    hybrid_accuracy - semantic_accuracy
)

reranker_improvement = (
    reranker_accuracy - hybrid_accuracy
)

print(
    f"\nHybrid'ın Semantic'e göre "
    f"doğruluk farkı : "
    f"{hybrid_improvement:+.2f} puan"
)

print(
    f"Reranker'ın Hybrid'e göre "
    f"doğruluk farkı : "
    f"{reranker_improvement:+.2f} puan"
)

print("\n" + "=" * 70)
print("ANALİZ TAMAMLANDI")
print("=" * 70)