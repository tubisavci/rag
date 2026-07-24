"""FAISS index kaydetme ve yeniden yükleme testi."""

# pylint: disable=no-value-for-parameter

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

INDEX_PATH = VECTOR_DB_DIR / "test_semantic.index"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TEST DOKÜMANLARI
# --------------------------------------------------

documents = [
    "Türkiye'nin uzay çalışmaları TÜBİTAK UZAY tarafından desteklenmektedir.",
    "Yapay zekâ teknolojileri sağlık ve eğitim alanlarında kullanılmaktadır.",
    "İklim değişikliği tarımsal üretim üzerinde önemli etkiler oluşturmaktadır.",
    "Siber güvenlik, bilgi sistemlerinin saldırılara karşı korunmasını amaçlar.",
    "Tüketici hakları, vatandaşların ekonomik çıkarlarının korunmasını sağlar.",
]


print("=" * 60)
print("FAISS INDEX KAYDETME / YÜKLEME TESTİ")
print("=" * 60)


# --------------------------------------------------
# 1. EMBEDDING MODELİNİ YÜKLE
# --------------------------------------------------

print(f"\nModel yükleniyor: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model başarıyla yüklendi.")


# --------------------------------------------------
# 2. DOKÜMAN EMBEDDINGLERİNİ OLUŞTUR
# --------------------------------------------------

print("\nDoküman embeddingleri oluşturuluyor...")

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=False,
).astype("float32")

faiss.normalize_L2(document_embeddings)

dimension = document_embeddings.shape[1]

print(f"Doküman sayısı: {len(documents)}")
print(f"Embedding boyutu: {dimension}")


# --------------------------------------------------
# 3. FAISS INDEX OLUŞTUR
# --------------------------------------------------

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print(f"Indexteki vektör sayısı: {index.ntotal}")


# --------------------------------------------------
# 4. DOSYA YOLUNU KONTROL ET
# --------------------------------------------------

print("\nDosya yolu kontrolü:")

print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
print(f"VECTOR_DB_DIR : {VECTOR_DB_DIR}")
print(f"Klasör mevcut : {VECTOR_DB_DIR.exists()}")
print(f"INDEX_PATH    : {INDEX_PATH}")


# --------------------------------------------------
# 5. INDEXİ SERIALIZE ET VE PYTHON İLE KAYDET
# --------------------------------------------------

print("\nFAISS index serialize ediliyor...")

serialized_index = faiss.serialize_index(index)

print(f"Serialize edilen veri boyutu: {serialized_index.nbytes} byte")


print("\nIndex Python ile diske yazılıyor...")

with INDEX_PATH.open("wb") as file:
    file.write(serialized_index.tobytes())

print("Index başarıyla diske kaydedildi.")
print(f"Kaydedilen dosya: {INDEX_PATH}")


# --------------------------------------------------
# 6. DOSYANIN OLUŞTUĞUNU KONTROL ET
# --------------------------------------------------

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index dosyası oluşturulamadı: {INDEX_PATH}"
    )

print(f"Index dosyası mevcut: {INDEX_PATH.exists()}")
print(f"Dosya boyutu: {INDEX_PATH.stat().st_size} byte")


# --------------------------------------------------
# 7. INDEX DOSYASINI PYTHON İLE OKU
# --------------------------------------------------

print("\nIndex dosyası diskten okunuyor...")

with INDEX_PATH.open("rb") as file:
    index_bytes = file.read()

serialized_loaded_index = np.frombuffer(
    index_bytes,
    dtype="uint8",
)


# --------------------------------------------------
# 8. FAISS INDEXİ YENİDEN OLUŞTUR
# --------------------------------------------------

loaded_index = faiss.deserialize_index(
    serialized_loaded_index
)

print("Index başarıyla tekrar yüklendi.")

print(
    f"Yüklenen indexteki vektör sayısı: "
    f"{loaded_index.ntotal}"
)


# --------------------------------------------------
# 9. TEST SORGUSU
# --------------------------------------------------

QUERY = (
    "Türkiye'de uzay araştırmaları hangi kurum "
    "tarafından destekleniyor?"
)

print(f"\nSorgu: {QUERY}")


# --------------------------------------------------
# 10. SORGU EMBEDDINGİNİ OLUŞTUR
# --------------------------------------------------

query_embedding = model.encode(
    [QUERY],
    convert_to_numpy=True,
    show_progress_bar=False,
).astype("float32")

faiss.normalize_L2(query_embedding)


# --------------------------------------------------
# 11. YÜKLENEN INDEX ÜZERİNDE ARAMA YAP
# --------------------------------------------------

scores, indices = loaded_index.search(
    query_embedding,
    TOP_K,
)


# --------------------------------------------------
# 12. SONUÇLARI GÖSTER
# --------------------------------------------------

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
print("FAISS persistence testi başarıyla tamamlandı.")
print("=" * 60)