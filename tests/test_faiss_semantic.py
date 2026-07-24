import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Test için kullanılacak örnek Türkçe metinler
documents = [
    "Türkiye'nin uzay çalışmaları TÜBİTAK UZAY tarafından desteklenmektedir.",
    "Yapay zekâ teknolojileri sağlık ve eğitim alanlarında kullanılmaktadır.",
    "İklim değişikliği tarımsal üretim üzerinde önemli etkiler oluşturmaktadır.",
    "Siber güvenlik, bilgi sistemlerinin saldırılara karşı korunmasını amaçlar.",
    "Tüketici hakları, vatandaşların ekonomik çıkarlarının korunmasını sağlar."
]


print("Embedding modeli yükleniyor...")
model = SentenceTransformer(MODEL_NAME)


# Dokümanları embedding'e dönüştür
document_embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")


print(f"Embedding shape: {document_embeddings.shape}")


# Cosine similarity için vektörleri normalize et
faiss.normalize_L2(document_embeddings)


# Embedding boyutunu al
dimension = document_embeddings.shape[1]


# Inner Product index oluştur
index = faiss.IndexFlatIP(dimension)


# Embeddingleri FAISS'e ekle
index.add(document_embeddings)

print(f"FAISS indexindeki vektör sayısı: {index.ntotal}")


# Kullanıcı sorgusu
query = "Türkiye'de uzay araştırmaları hangi kurum tarafından destekleniyor?"


query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")


# Query vektörünü de normalize et
faiss.normalize_L2(query_embedding)


# En alakalı 3 sonucu getir
k = 3

scores, indices = index.search(query_embedding, k)


print(f"\nSorgu: {query}")
print("\nEn alakalı sonuçlar:")


for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    print(f"\n{rank}. Sonuç")
    print(f"Benzerlik skoru: {score:.4f}")
    print(f"Metin: {documents[idx]}")