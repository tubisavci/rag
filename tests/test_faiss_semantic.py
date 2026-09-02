"""SentenceTransformer ve FAISS ile semantic search testi."""

# pylint: disable=no-value-for-parameter

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3

# Test için kullanılacak Türkçe dokümanlar
documents = [
    "Türkiye'nin uzay çalışmaları TÜBİTAK UZAY tarafından desteklenmektedir.",
    "Yapay zekâ teknolojileri sağlık ve eğitim alanlarında kullanılmaktadır.",
    "İklim değişikliği tarımsal üretim üzerinde önemli etkiler oluşturmaktadır.",
    "Siber güvenlik, bilgi sistemlerinin saldırılara karşı korunmasını amaçlar.",
    "Tüketici hakları, vatandaşların ekonomik çıkarlarının korunmasını sağlar.",
]

print("=" * 60)
print("FAISS SEMANTIC SEARCH TESTİ")
print("=" * 60)

# Embedding modelini yükle
print(f"\nModel yükleniyor: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model başarıyla yüklendi.")

# Doküman embeddinglerini oluştur
print("\nDoküman embeddingleri oluşturuluyor...")

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=False,
).astype("float32")

print(f"Doküman sayısı: {len(documents)}")
print(f"Embedding shape: {document_embeddings.shape}")

# Cosine similarity için embeddingleri normalize et
faiss.normalize_L2(document_embeddings)

# FAISS index oluştur
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

print(f"Embedding boyutu: {dimension}")
print(f"FAISS indexindeki vektör sayısı: {index.ntotal}")

# Kullanıcı sorgusu
QUERY = "Türkiye'de uzay araştırmaları hangi kurum tarafından destekleniyor?"

print(f"\nSorgu: {QUERY}")

# Sorguyu embedding'e dönüştür
query_embedding = model.encode(
    [QUERY],
    convert_to_numpy=True,
    show_progress_bar=False,
).astype("float32")

# Cosine similarity için sorgu embeddingini normalize et
faiss.normalize_L2(query_embedding)

# Semantic search gerçekleştir
scores, indices = index.search(query_embedding, TOP_K)

# Sonuçları göster
print(f"\nTop-{TOP_K} sonuç:")
print("-" * 60)

for rank, (idx, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1,
):
    print(f"\n{rank}. Sonuç")
    print(f"Doküman indeksi : {idx}")
    print(f"Benzerlik skoru : {score:.4f}")
    print(f"Metin            : {documents[idx]}")

print("\n" + "=" * 60)
print("Semantic search testi tamamlandı.")
print("=" * 60)