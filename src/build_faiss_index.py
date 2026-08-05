"""
BGE-M3 kullanarak chunk embeddingleri ve FAISS index oluşturur.

Kullanım:

python src/build_faiss_index.py --strategy 300_50
python src/build_faiss_index.py --strategy 500_100
python src/build_faiss_index.py --strategy 800_150
"""

import argparse
import json
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 16


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():
    """Komut satırı argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description="BGE-M3 ile FAISS index oluştur."
    )

    parser.add_argument(
        "--strategy",
        default="300_50",
        choices=[
            "300_50",
            "500_100",
            "800_150",
        ],
        help="Chunk stratejisi",
    )

    return parser.parse_args()


# --------------------------------------------------
# PATHLER
# --------------------------------------------------

def get_paths(strategy):
    """Proje yollarını oluşturur."""

    project_root = Path(__file__).resolve().parent.parent

    chunk_path = (
        project_root
        / "data"
        / "chunks"
        / f"chunks_{strategy}.json"
    )

    vector_db_dir = (
        project_root
        / "vector_db"
    )

    vector_db_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        vector_db_dir
        / f"faiss_bge_m3_{strategy}.index"
    )

    metadata_path = (
        vector_db_dir
        / f"metadata_bge_m3_{strategy}.json"
    )

    return (
        project_root,
        chunk_path,
        vector_db_dir,
        index_path,
        metadata_path,
    )


# --------------------------------------------------
# CHUNKLARI YÜKLE
# --------------------------------------------------

def load_chunks(chunk_path):
    """Chunk dosyasını yükler."""

    print("\nChunk dosyası yükleniyor...")

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunk dosyası bulunamadı:\n{chunk_path}"
        )

    with chunk_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError(
            "Chunk dosyası boş."
        )

    print(
        f"Toplam chunk sayısı: {len(chunks)}"
    )

    return chunks


# --------------------------------------------------
# METİNLERİ AYIR
# --------------------------------------------------

def extract_texts(chunks):
    """Chunklardan metinleri çıkarır."""

    texts = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            "",
        ).strip()

        if not text:

            raise ValueError(
                f"Boş chunk bulundu. "
                f"Chunk ID: {chunk.get('chunk_id')}"
            )

        texts.append(text)

    print(
        f"Embedding üretilecek metin sayısı: "
        f"{len(texts)}"
    )

    return texts


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_model():
    """BGE-M3 modelini yükler."""

    print("\nBGE-M3 yükleniyor...")

    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    load_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Model başarıyla yüklendi. "
        f"Süre: {load_time:.2f} saniye"
    )

    return (
        model,
        load_time,
    )


# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------

def build_embeddings(
    model,
    texts,
):
    """Embedding üretir."""

    print(
        "\nChunk embeddingleri oluşturuluyor..."
    )

    print(
        "Bu işlem CPU üzerinde biraz sürebilir.\n"
    )

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(
        "float32"
    )

    embedding_time = (
        time.perf_counter()
        - start
    )

    print(
        "\nEmbedding işlemi tamamlandı."
    )

    print(
        f"Embedding shape : "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding süresi: "
        f"{embedding_time:.2f} saniye"
    )

    return (
        embeddings,
        embedding_time,
    )


# --------------------------------------------------
# FAISS
# --------------------------------------------------

def build_index(
    embeddings,
    chunk_count,
):
    """FAISS index oluşturur."""

    dimension = embeddings.shape[1]

    print(
        "\nFAISS index oluşturuluyor..."
    )

    print(
        f"Embedding boyutu: "
        f"{dimension}"
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS indexindeki vektör sayısı: "
        f"{index.ntotal}"
    )

    if index.ntotal != chunk_count:

        raise ValueError(
            "FAISS vektör sayısı ile "
            "chunk sayısı eşleşmiyor."
        )

    return (
        index,
        dimension,
    )


# --------------------------------------------------
# SERIALIZE
# --------------------------------------------------

def save_index(
    index,
    index_path,
):
    """FAISS indexi kaydeder."""

    print(
        "\nFAISS index serialize ediliyor..."
    )

    serialized_index = (
        faiss.serialize_index(
            index
        )
    )

    print(
        f"Serialize edilen index boyutu: "
        f"{serialized_index.nbytes / (1024 * 1024):.2f} MB"
    )

    print(
        "\nFAISS index diske kaydediliyor..."
    )

    with index_path.open(
        "wb"
    ) as file:

        file.write(
            serialized_index.tobytes()
        )

    print(
        "FAISS index başarıyla kaydedildi."
    )

    # --------------------------------------------------
# METADATA
# --------------------------------------------------

def save_metadata(
    chunks,
    metadata_path,
):
    """Metadata dosyasını kaydeder."""

    print(
        "\nMetadata kaydediliyor..."
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Metadata başarıyla kaydedildi."
    )


# --------------------------------------------------
# DOSYA KONTROLÜ
# --------------------------------------------------

def verify_files(
    index_path,
    metadata_path,
):
    """Oluşturulan dosyaları kontrol eder."""

    if not index_path.exists():

        raise FileNotFoundError(
            f"FAISS index oluşturulamadı:\n{index_path}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata oluşturulamadı:\n{metadata_path}"
        )


# --------------------------------------------------
# RAPOR
# --------------------------------------------------

def print_summary(
    strategy,
    chunk_count,
    dimension,
    load_time,
    embedding_time,
    index_path,
    metadata_path,
):
    """İşlem özetini yazdırır."""

    index_size_mb = (
        index_path.stat().st_size
        / (1024 * 1024)
    )

    metadata_size_mb = (
        metadata_path.stat().st_size
        / (1024 * 1024)
    )

    average_embedding_time = (
        embedding_time
        / chunk_count
    )

    print("\n" + "=" * 70)
    print("INDEX OLUŞTURMA TAMAMLANDI")
    print("=" * 70)

    print(
        f"Chunk stratejisi        : {strategy}"
    )

    print(
        f"Toplam chunk            : {chunk_count}"
    )

    print(
        f"Embedding boyutu        : {dimension}"
    )

    print(
        f"Model yükleme süresi    : "
        f"{load_time:.2f} sn"
    )

    print(
        f"Toplam embedding süresi : "
        f"{embedding_time:.2f} sn"
    )

    print(
        f"Ort. chunk embedding    : "
        f"{average_embedding_time:.4f} sn"
    )

    print(
        f"FAISS index boyutu      : "
        f"{index_size_mb:.2f} MB"
    )

    print(
        f"Metadata boyutu         : "
        f"{metadata_size_mb:.2f} MB"
    )

    print("\nOluşturulan dosyalar:")

    print(index_path)
    print(metadata_path)

    print("=" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    """Programın ana giriş noktası."""

    args = parse_arguments()

    (
        _,
        chunk_path,
        _,
        index_path,
        metadata_path,
    ) = get_paths(
        args.strategy
    )

    print("=" * 70)
    print("BGE-M3 + FAISS INDEX OLUŞTURMA")
    print("=" * 70)

    print(f"\nModel           : {MODEL_NAME}")
    print(f"Chunk stratejisi: {args.strategy}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Chunk dosyası   : {chunk_path}")
    print(f"FAISS index     : {index_path}")
    print(f"Metadata        : {metadata_path}")

    chunks = load_chunks(
        chunk_path
    )

    texts = extract_texts(
        chunks
    )

    (
        model,
        load_time,
    ) = load_model()

    (
        embeddings,
        embedding_time,
    ) = build_embeddings(
        model,
        texts,
    )

    (
        index,
        dimension,
    ) = build_index(
        embeddings,
        len(chunks),
    )

    save_index(
        index,
        index_path,
    )

    save_metadata(
        chunks,
        metadata_path,
    )

    verify_files(
        index_path,
        metadata_path,
    )

    print_summary(
        strategy=args.strategy,
        chunk_count=len(chunks),
        dimension=dimension,
        load_time=load_time,
        embedding_time=embedding_time,
        index_path=index_path,
        metadata_path=metadata_path,
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()