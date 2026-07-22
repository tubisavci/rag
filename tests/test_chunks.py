import json
from pathlib import Path


# ---------------------------------------------------------
# TEST EDİLECEK CHUNK STRATEJİLERİ
# ---------------------------------------------------------

STRATEGIES = [
    {
        "file": Path("data/chunks/chunks_300_50.json"),
        "chunk_size": 300,
        "overlap": 50
    },
    {
        "file": Path("data/chunks/chunks_500_100.json"),
        "chunk_size": 500,
        "overlap": 100
    },
    {
        "file": Path("data/chunks/chunks_800_150.json"),
        "chunk_size": 800,
        "overlap": 150
    }
]

EXPECTED_SOURCE_COUNT = 10


# ---------------------------------------------------------
# STRATEJİ TESTİ
# ---------------------------------------------------------

def test_strategy(strategy):

    chunks_file = strategy["file"]
    max_token_count = strategy["chunk_size"]
    overlap = strategy["overlap"]

    print("\n" + "=" * 60)

    print(
        f"{max_token_count}/{overlap} "
        f"CHUNK KALİTE KONTROLÜ"
    )

    print("=" * 60)

    # JSON dosyasını oku.
    with chunks_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    # Kaynak belgeleri bul.
    sources = {
        chunk["source"]
        for chunk in chunks
    }

    # Token sayılarını al.
    token_counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    # Maksimum sınırı aşan chunk'lar.
    oversized_chunks = [
        chunk
        for chunk in chunks
        if chunk["token_count"] > max_token_count
    ]

    # Boş chunk'lar.
    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk["text"].strip()
    ]

    # İstatistikler.
    minimum = min(token_counts)
    maximum = max(token_counts)
    average = sum(token_counts) / len(token_counts)

    print(
        f"\nToplam chunk sayısı: "
        f"{len(chunks)}"
    )

    print(
        f"Kaynak belge sayısı: "
        f"{len(sources)}"
    )

    print(
        f"En küçük chunk: "
        f"{minimum} token"
    )

    print(
        f"En büyük chunk: "
        f"{maximum} token"
    )

    print(
        f"Ortalama chunk: "
        f"{average:.2f} token"
    )

    print(
        f"{max_token_count} tokenı aşan chunk: "
        f"{len(oversized_chunks)}"
    )

    print(
        f"Boş chunk: "
        f"{len(empty_chunks)}"
    )

    print("\n" + "-" * 60)

    if (
        len(sources) == EXPECTED_SOURCE_COUNT
        and len(oversized_chunks) == 0
        and len(empty_chunks) == 0
    ):
        print("SONUÇ: BAŞARILI ✅")

    else:
        print("SONUÇ: KONTROL GEREKİYOR ❌")

    print("-" * 60)


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("CHUNK STRATEJİLERİ KALİTE KONTROLÜ")
    print("=" * 60)

    for strategy in STRATEGIES:
        test_strategy(strategy)


if __name__ == "__main__":
    main()