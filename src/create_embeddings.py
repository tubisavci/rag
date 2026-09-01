import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from embedding_model import BGEEmbeddingModel


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_DIR = BASE_DIR / "data" / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

STRATEGIES = {
    "300_50": CHUNKS_DIR / "chunks_300_50.json",
    "500_100": CHUNKS_DIR / "chunks_500_100.json",
    "800_150": CHUNKS_DIR / "chunks_800_150.json",
}


# =========================================================
# ARGÜMAN
# =========================================================

def get_strategy():

    if len(sys.argv) != 3:

        print(
            "Kullanım: "
            "python src/create_embeddings.py "
            "--strategy 300_50"
        )

        raise SystemExit(1)

    if sys.argv[1] != "--strategy":

        print("Hata: --strategy kullanılmalı.")

        raise SystemExit(1)

    strategy = sys.argv[2]

    if strategy not in STRATEGIES:

        print(
            f"Hata: Geçersiz strateji: {strategy}"
        )

        print(
            "Geçerli stratejiler:"
        )

        for value in STRATEGIES:

            print(f"  - {value}")

        raise SystemExit(1)

    return strategy

# =========================================================
# CHUNK OKU
# =========================================================

def load_chunks(path):

    print("\nChunk dosyası yükleniyor...")
    print(f"Dosya: {path}")

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    print(
        f"Toplam chunk: {len(chunks)}"
    )

    return chunks


# =========================================================
# ANA PROGRAM
# =========================================================

def main():

    strategy = get_strategy()

    chunk_path = STRATEGIES[strategy]

    print("=" * 70)
    print("BGE-M3 EMBEDDING ÜRETİMİ")
    print("=" * 70)

    print(
        f"\nModel      : BAAI/bge-m3"
    )

    print(
        f"Strateji   : {strategy}"
    )

    print(
        f"Chunk dosyası : {chunk_path.name}"
    )

    if not chunk_path.exists():

        raise FileNotFoundError(
            f"Chunk dosyası bulunamadı: "
            f"{chunk_path}"
        )

    chunks = load_chunks(chunk_path)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = BGEEmbeddingModel()

    print("\nEmbedding üretimi başlıyor...")

    start = time.perf_counter()

    embeddings = model.encode(
        texts
    )

    elapsed = time.perf_counter() - start

    print(
        f"\nEmbedding üretimi tamamlandı."
    )

    print(
        f"Süre        : {elapsed:.2f} sn"
    )

    print(
        f"Shape       : {embeddings.shape}"
    )

    print(
        f"Dtype       : {embeddings.dtype}"
    )

    # -----------------------------------------------------
    # KONTROLLER
    # -----------------------------------------------------

    if len(embeddings) != len(chunks):

        raise ValueError(
            "Embedding ve chunk sayıları "
            "eşleşmiyor!"
        )

    if embeddings.dtype != np.float32:

        embeddings = embeddings.astype(
            "float32"
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise ValueError(
            "Embedding içerisinde "
            "geçersiz NaN/Inf değerleri var!"
        )

    # -----------------------------------------------------
    # KAYDET
    # -----------------------------------------------------

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        EMBEDDINGS_DIR
        / f"embeddings_{strategy}.npy"
    )

    np.save(
        output_path,
        embeddings
    )

    # -----------------------------------------------------
    # SONUÇ
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("EMBEDDING KAYDEDİLDİ")
    print("=" * 70)

    print(
        f"\nDosya       : {output_path}"
    )

    print(
        f"Embedding   : {len(embeddings)}"
    )

    print(
        f"Boyut       : {embeddings.shape[1]}"
    )

    if torch.cuda.is_available():

        print(
            f"VRAM        : "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    print(
        f"Süre        : {elapsed:.2f} sn"
    )


if __name__ == "__main__":
    main()