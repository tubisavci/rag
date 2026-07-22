import json
from pathlib import Path


CHUNKS_FILE = Path("data/chunks/chunks_300_50.json")
MAX_TOKEN_COUNT = 300
EXPECTED_SOURCE_COUNT = 10


def main():

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print("=" * 60)
    print("300/50 CHUNK KALİTE KONTROLÜ")
    print("=" * 60)

    print(f"\nToplam chunk sayısı: {len(chunks)}")

    sources = {
        chunk["source"]
        for chunk in chunks
    }

    print(f"Kaynak belge sayısı: {len(sources)}")

    token_counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    print(f"En küçük chunk: {min(token_counts)} token")
    print(f"En büyük chunk: {max(token_counts)} token")

    average = sum(token_counts) / len(token_counts)

    print(f"Ortalama chunk: {average:.2f} token")

    oversized_chunks = [
        chunk
        for chunk in chunks
        if chunk["token_count"] > MAX_TOKEN_COUNT
    ]

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk["text"].strip()
    ]

    print(
        f"300 tokenı aşan chunk: "
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


if __name__ == "__main__":
    main()