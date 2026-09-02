"""
FAISS Semantic Search

Kullanım:

python src/search_faiss.py --strategy 300_50

python src/search_faiss.py --strategy 500_100

python src/search_faiss.py --strategy 800_150
"""

import argparse
import json
import time
from pathlib import Path

from embedding_model import BGEEmbeddingModel

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():
    """Komut satırı argümanlarını okur."""

    parser = argparse.ArgumentParser(
        description="FAISS Semantic Search"
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

    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help="Getirilecek sonuç sayısı",
    )

    return parser.parse_args()


# --------------------------------------------------
# PATHLER
# --------------------------------------------------

def get_paths(strategy):
    """Dosya yollarını oluşturur."""

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    index_path = (
        project_root
        / "vector_db"
        / f"faiss_bge_m3_{strategy}.index"
    )

    metadata_path = (
        project_root
        / "vector_db"
        / f"metadata_bge_m3_{strategy}.json"
    )

    return (
        index_path,
        metadata_path,
    )


# --------------------------------------------------
# INDEX
# --------------------------------------------------

def load_index(index_path):
    """FAISS indexini yükler."""

    import faiss
    import numpy as np

    print("\nFAISS index yükleniyor...")

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    with index_path.open("rb") as file:

        serialized = np.frombuffer(
            file.read(),
            dtype="uint8",
        )

    index = faiss.deserialize_index(
        serialized
    )

    print(
        "Index başarıyla yüklendi."
    )

    print(
        f"Toplam vektör: {index.ntotal}"
    )

    return index


# --------------------------------------------------
# METADATA
# --------------------------------------------------

def load_metadata(metadata_path):
    """Metadata dosyasını yükler."""

    print("\nMetadata yükleniyor...")

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    print(
        f"Toplam metadata: {len(metadata)}"
    )

    return metadata


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_model():
    """BGE-M3 embedding modelini yükler."""

    print(
        "\nBGE-M3 embedding modeli "
        "yükleniyor..."
    )

    start = time.perf_counter()

    model = BGEEmbeddingModel()

    load_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Model hazır "
        f"({load_time:.2f} sn)"
    )

    return (
        model,
        load_time,
    )


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def embed_query(
    model,
    question,
):
    """Sorguyu BGE-M3 embeddingine dönüştürür."""

    embedding = model.encode(
        [question],
        batch_size=1,
    ).astype(
        "float32"
    )

    return embedding


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

def search(
    index,
    query_embedding,
    top_k,
):
    """FAISS üzerinde semantic search yapar."""

    start = time.perf_counter()

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        scores,
        indices,
        elapsed,
    )


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

def print_results(
    question,
    metadata,
    scores,
    indices,
    search_time,
):
    """Semantic search sonuçlarını yazdırır."""

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH SONUCU")
    print("=" * 70)

    print(
        f"\nSoru: {question}"
    )

    print(
        f"\nToplam sonuç: "
        f"{len(indices[0])}"
    )

    print(
        f"Arama süresi: "
        f"{search_time:.4f} saniye"
    )

    print("\n" + "=" * 70)

    for rank, (idx, score) in enumerate(
        zip(
            indices[0],
            scores[0],
        ),
        start=1,
    ):

        # FAISS bazı durumlarda -1
        # döndürebilir.
        if idx < 0:
            continue

        chunk = metadata[int(idx)]

        print(
            f"\n{rank}. SONUÇ"
        )

        print(
            f"Benzerlik Skoru : "
            f"{score:.4f}"
        )

        print(
            f"Kaynak          : "
            f"{chunk['source']}"
        )

        print(
            f"Chunk ID        : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Chunk Index     : "
            f"{chunk['chunk_index']}"
        )

        print(
            f"Token Sayısı    : "
            f"{chunk['token_count']}"
        )

        print("-" * 70)

        text = chunk["text"].strip()

        if len(text) > 700:
            text = (
                text[:700]
                + " ..."
            )

        print(text)

        print("-" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_arguments()

    index_path, metadata_path = (
        get_paths(
            args.strategy
        )
    )

    print("=" * 70)
    print("FAISS SEMANTIC SEARCH")
    print("=" * 70)

    print(
        f"\nModel            : "
        f"{MODEL_NAME}"
    )

    print(
        f"Chunk Stratejisi : "
        f"{args.strategy}"
    )

    print(
        f"Top-K            : "
        f"{args.top_k}"
    )

    print(
        f"Index            : "
        f"{index_path}"
    )

    print(
        f"Metadata         : "
        f"{metadata_path}"
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    # ÖNEMLİ:
    # BGE-M3 modeli FAISS indexinden
    # önce yükleniyor.
    #
    # Windows'taki native kütüphane
    # çakışmasını önlemek için bu sıra
    # korunmalıdır.

    model, model_time = load_model()

    print(
        f"\nModel yükleme süresi: "
        f"{model_time:.2f} sn"
    )

    # --------------------------------------------------
    # FAISS INDEX
    # --------------------------------------------------

    index = load_index(
        index_path
    )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    metadata = load_metadata(
        metadata_path
    )

    # --------------------------------------------------
    # KONTROLLER
    # --------------------------------------------------

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS index ve metadata "
            "sayıları eşleşmiyor!"
        )

    print(
        "\nFAISS index ve metadata "
        "kontrolü başarılı."
    )

    # --------------------------------------------------
    # SORU DÖNGÜSÜ
    # --------------------------------------------------

    while True:

        print(
            "\nÇıkmak için 'q' "
            "yazabilirsiniz."
        )

        question = input(
            "\nSorunuzu girin:\n> "
        ).strip()

        if question.lower() == "q":

            print(
                "\nProgram sonlandırıldı."
            )

            break

        if question == "":

            print(
                "\nBoş soru giremezsiniz."
            )

            continue

        try:

            # --------------------------------------------------
            # QUERY EMBEDDING
            # --------------------------------------------------

            query_embedding = embed_query(
                model,
                question,
            )

            # --------------------------------------------------
            # SEMANTIC SEARCH
            # --------------------------------------------------

            (
                scores,
                indices,
                search_time,
            ) = search(
                index,
                query_embedding,
                args.top_k,
            )

            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            print_results(
                question,
                metadata,
                scores,
                indices,
                search_time,
            )

        except Exception as exc:

            print(
                "\nARAMA SIRASINDA HATA:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()