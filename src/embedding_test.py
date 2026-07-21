from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Kullanılacak embedding modeli
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


print("Embedding modeli yükleniyor...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding modeli başarıyla yüklendi.")


# ==================================================
# 1. EMBEDDING OLUŞTURMA TESTİ
# ==================================================

# Türkçe test cümlesi
sentence = "Yapay zeka geleceğin teknolojisidir."

# Cümleyi embedding vektörüne dönüştür
embedding = model.encode(sentence)

print("\nTest cümlesi:")
print(sentence)

print("\nEmbedding boyutu:")
print(embedding.shape)

print("\nEmbedding vektörünün ilk 10 değeri:")
print(embedding[:10])


# ==================================================
# 2. ANLAMSAL BENZERLİK TESTİ
# ==================================================

print("\n" + "=" * 50)
print("ANLAMSAL BENZERLİK TESTİ")
print("=" * 50)

sentence1 = "Yapay zeka geleceğin teknolojisidir."

sentence2 = (
    "Yapay zeka gelecekte önemli bir teknoloji olacaktır."
)

sentence3 = "Bugün akşam makarna yapacağım."


# Cümlelerin embedding vektörlerini oluştur
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)


# Cosine similarity hesapla
similarity_1_2 = cos_sim(
    embedding1,
    embedding2
).item()

similarity_1_3 = cos_sim(
    embedding1,
    embedding3
).item()


# Cümleleri ekrana yazdır
print("\nCümle 1:")
print(sentence1)

print("\nCümle 2:")
print(sentence2)

print("\nCümle 3:")
print(sentence3)


# Benzerlik sonuçlarını göster
print("\nBenzerlik sonuçları:")

print(
    f"Cümle 1 - Cümle 2 benzerliği: "
    f"{similarity_1_2:.4f}"
)

print(
    f"Cümle 1 - Cümle 3 benzerliği: "
    f"{similarity_1_3:.4f}"
)