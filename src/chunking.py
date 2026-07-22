import json
from pathlib import Path

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CLEAN_DATA_DIR = Path("data/clean")
CHUNKS_DATA_DIR = Path("data/chunks")

# Gün 13 chunking stratejisi
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Ayarlara göre çıktı dosyasının adı otomatik oluşturulur.
OUTPUT_FILE = (
    CHUNKS_DATA_DIR
    / f"chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}.json"
)


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

print("Tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer başarıyla yüklendi.")


def token_length(text):
    """
    Metnin embedding modelinin tokenizer'ına göre
    token sayısını hesaplar.
    """

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# ---------------------------------------------------------
# TEXT SPLITTER
# ---------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_length,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

def main():

    # Çıktı klasörü yoksa oluştur.
    CHUNKS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Temizlenmiş TXT belgelerini bul.
    txt_files = sorted(
        CLEAN_DATA_DIR.glob("*.txt")
    )

    print("\n" + "=" * 60)

    print(
        f"{CHUNK_SIZE} TOKEN / "
        f"{CHUNK_OVERLAP} OVERLAP CHUNKING"
    )

    print("=" * 60)

    print(
        f"\nBulunan TXT sayısı: "
        f"{len(txt_files)}"
    )

    all_chunks = []

    global_chunk_id = 0

    # -----------------------------------------------------
    # BELGELERİ CHUNK'LARA AYIR
    # -----------------------------------------------------

    for txt_path in txt_files:

        print("\n" + "-" * 60)

        print(
            f"İşleniyor: "
            f"{txt_path.name}"
        )

        text = txt_path.read_text(
            encoding="utf-8"
        )

        chunks = splitter.split_text(
            text
        )

        print(
            f"Oluşturulan chunk sayısı: "
            f"{len(chunks)}"
        )

        # -------------------------------------------------
        # CHUNK METADATA
        # -------------------------------------------------

        for chunk_index, chunk in enumerate(chunks):

            chunk_data = {
                "chunk_id": global_chunk_id,
                "source": txt_path.name,
                "chunk_index": chunk_index,
                "token_count": token_length(chunk),
                "text": chunk
            }

            all_chunks.append(
                chunk_data
            )

            global_chunk_id += 1

    # -----------------------------------------------------
    # JSON OLARAK KAYDET
    # -----------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    # -----------------------------------------------------
    # SONUÇ
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("CHUNKING TAMAMLANDI")

    print("=" * 60)

    print(
        f"\nToplam chunk sayısı: "
        f"{len(all_chunks)}"
    )

    print(
        f"Çıktı dosyası: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()