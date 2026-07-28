"""Multilingual E5 Base retrieval benchmark testi."""

import time

import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 3


# --------------------------------------------------
# TEST DOKÜMANLARI
# --------------------------------------------------

raw_documents = [
    "Türkiye'nin uzay çalışmaları TÜBİTAK UZAY tarafından desteklenmektedir.",
    "Yapay zekâ teknolojileri sağlık ve eğitim alanlarında kullanılmaktadır.",
    "İklim değişikliği tarımsal üretimi, su kaynaklarını ve gıda güvenliğini etkileyebilir.",
    "Siber güvenlik, bilgi sistemlerinin saldırılara ve yetkisiz erişime karşı korunmasını amaçlar.",
    "Tüketici hakları, vatandaşların ekonomik çıkarlarının ve güvenliğinin korunmasını kapsar.",
]

# E5 retrieval formatı
documents = [
    f"passage: {document}"
    for document in raw_documents
]


# --------------------------------------------------
# TEST SORGULARI
# --------------------------------------------------

queries = [
    {
        "query": "Türkiye'deki uzay çalışmalarını hangi kurum desteklemektedir?",
        "expected_index": 0,
    },
    {
        "query": "Sağlık ve eğitim alanlarında kullanılan teknoloji hangisidir?",
        "expected_index": 1,
    },
    {
        "query": "İklim değişikliği tarım ve gıda güvenliğini nasıl etkiler?",
        "expected_index": 2,
    },
    {
        "query": "Bilgi sistemlerini saldırılardan koruyan alan nedir?",
        "expected_index": 3,
    },
    {
        "query": "Vatandaşların ekonomik çıkarlarını koruyan haklar nelerdir?",
        "expected_index": 4,
    },
]


print("=" * 70)
print("MULTILINGUAL-E5-BASE RETRIEVAL BENCHMARK")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")


# --------------------------------------------------
# 1. MODELİ YÜKLE
# --------------------------------------------------

start_time = time.perf_counter()

model = SentenceTransformer(MODEL_NAME)

model_load_time = time.perf_counter() - start_time

print(f"Model yükleme süresi: {model_load_time:.4f} saniye")


# --------------------------------------------------
# 2. DOKÜMAN EMBEDDINGLERİNİ OLUŞTUR
# --------------------------------------------------

start_time = time.perf_counter()

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=False,
).astype("float32")

document_encoding_time = time.perf_counter() - start_time

dimension = document_embeddings.shape[1]

print(f"Doküman embedding süresi: {document_encoding_time:.4f} saniye")
print(f"Embedding boyutu: {dimension}")


# --------------------------------------------------
# 3. FAISS INDEX
# --------------------------------------------------

faiss.normalize_L2(document_embeddings)

index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)


# --------------------------------------------------
# 4. SORGULARI TEST ET
# --------------------------------------------------

correct_top1 = 0
total_query_time = 0.0

print("\n" + "-" * 70)
print("SORGULAR")
print("-" * 70)

for number, item in enumerate(queries, start=1):

    raw_query = item["query"]
    expected_index = item["expected_index"]

    # E5 query formatı
    query = f"query: {raw_query}"

    start_time = time.perf_counter()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    query_time = time.perf_counter() - start_time
    total_query_time += query_time

    predicted_index = int(indices[0][0])

    is_correct = predicted_index == expected_index

    if is_correct:
        correct_top1 += 1

    print(f"\n{number}. Sorgu: {raw_query}")
    print(f"Beklenen doküman : {expected_index}")
    print(f"Top-1 doküman    : {predicted_index}")
    print(f"Top-1 skor       : {scores[0][0]:.4f}")
    print(f"Doğru mu?        : {'EVET' if is_correct else 'HAYIR'}")
    print(f"Sorgu süresi     : {query_time:.4f} saniye")


# --------------------------------------------------
# 5. ÖZET
# --------------------------------------------------

accuracy = correct_top1 / len(queries)
average_query_time = total_query_time / len(queries)

print("\n" + "=" * 70)
print("BENCHMARK SONUCU")
print("=" * 70)

print(f"Model                 : {MODEL_NAME}")
print(f"Embedding boyutu      : {dimension}")
print(f"Model yükleme süresi  : {model_load_time:.4f} saniye")
print(f"Doküman encode süresi : {document_encoding_time:.4f} saniye")
print(f"Ort. sorgu süresi     : {average_query_time:.4f} saniye")
print(
    f"Top-1 doğruluk        : "
    f"{correct_top1}/{len(queries)} ({accuracy * 100:.2f}%)"
)

print("=" * 70)