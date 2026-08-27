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


print("=" * 70)
print("PERFORMANS ANALİZİ")
print("=" * 70)

print(f"\nToplam test sorusu : {len(results)}")

# --------------------------------------------------
# ORTALAMA SÜRELER
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



semantic_avg = sum(semantic_times) / len(semantic_times)
bm25_avg = sum(bm25_times) / len(bm25_times)
hybrid_avg = sum(hybrid_times) / len(hybrid_times)

print("\nORTALAMA ARAMA SÜRELERİ")
print("-" * 30)

print(f"Semantic           : {semantic_avg:.4f} sn")
print(f"BM25               : {bm25_avg:.4f} sn")
print(f"Hybrid             : {hybrid_avg:.4f} sn")
print(f"Hybrid + Reranker  : {reranker_avg:.4f} sn")

# --------------------------------------------------
# ORTALAMA SÜRELER
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


semantic_avg = sum(semantic_times) / len(semantic_times)
bm25_avg = sum(bm25_times) / len(bm25_times)
hybrid_avg = sum(hybrid_times) / len(hybrid_times)


print("\nORTALAMA ARAMA SÜRELERİ")
print("-" * 30)

print(f"Semantic : {semantic_avg:.4f} sn")
print(f"BM25     : {bm25_avg:.4f} sn")
print(f"Hybrid   : {hybrid_avg:.4f} sn")

