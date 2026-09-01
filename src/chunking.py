import json
from pathlib import Path

from transformers import AutoTokenizer


# =========================================================
# AYARLAR
# =========================================================

MODEL_NAME = "BAAI/bge-m3"

MODEL_PATH = Path(
    r"C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-m3"
    r"\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)

CLEAN_DATA_DIR = Path("data/clean")
CHUNKS_DATA_DIR = Path("data/chunks")


# =========================================================
# CHUNK STRATEJİSİ
# =========================================================

# 37. Gün deneylerinde şu üç strateji kullanılacak:
#
# 300 / 50
# 500 / 100
# 800 / 150
#
# Şu anda 800 / 150 üretiliyor.

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


OUTPUT_FILE = (
    CHUNKS_DATA_DIR
    / f"chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}.json"
)


# =========================================================
# TOKENIZER
# =========================================================

print("BGE-M3 tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

# ---------------------------------------------------------
# ÖNEMLİ
# ---------------------------------------------------------
#
# Uzun belgeleri tokenizer ile işlerken Transformers,
# modelin maksimum sequence length değerini kontrol edip
# gereksiz bir uyarı yazabiliyor.
#
# Biz modeli çalıştırmıyoruz.
# Sadece tokenizer kullanıyoruz.
#
# Chunk'larımız zaten maksimum 800 token olacak.
#
# Bu nedenle tokenizer'ın uyarı sınırını yükseltiyoruz.
# Bu, modelin gerçek context limitini değiştirmez.
# ---------------------------------------------------------

tokenizer.model_max_length = 10_000_000


print("BGE-M3 tokenizer başarıyla yüklendi.")


# =========================================================
# METNİ TOKENLARA AYIR
# =========================================================

def tokenize_text(text):
    """
    Verilen metni BGE-M3 tokenizer kullanarak
    token ID listesine dönüştürür.

    Model çalıştırılmaz.
    Sadece tokenizer kullanılır.
    """

    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )


# =========================================================
# TOKEN TABANLI CHUNKING
# =========================================================

def split_text_by_tokens(
    text,
    chunk_size,
    chunk_overlap
):
    """
    Metni token tabanlı olarak chunk'lara ayırır.

    Örneğin:

        chunk_size = 500
        chunk_overlap = 100

    ise:

        Chunk 1 -> 0 - 500
        Chunk 2 -> 400 - 900
        Chunk 3 -> 800 - 1300

    şeklinde ilerler.

    Böylece iki ardışık chunk arasında 100 token
    overlap bulunur.
    """

    # -----------------------------------------------------
    # PARAMETRE KONTROLLERİ
    # -----------------------------------------------------

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size 0'dan büyük olmalıdır."
        )

    if chunk_overlap < 0:

        raise ValueError(
            "chunk_overlap negatif olamaz."
        )

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "chunk_overlap, chunk_size değerinden "
            "küçük olmalıdır."
        )

    # -----------------------------------------------------
    # TOKENIZE
    # -----------------------------------------------------

    token_ids = tokenize_text(text)

    if not token_ids:
        return []

    # -----------------------------------------------------
    # CHUNK'LARA AYIR
    # -----------------------------------------------------

    chunks = []

    step = chunk_size - chunk_overlap

    start = 0

    while start < len(token_ids):

        end = min(
            start + chunk_size,
            len(token_ids)
        )

        # Orijinal token listesinden doğrudan
        # ilgili tokenları al.
        chunk_token_ids = token_ids[start:end]

        # Tokenları tekrar metne dönüştür.
        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True
        ).strip()

        if chunk_text:

            chunks.append(
                {
                    "text": chunk_text,

                    # DİKKAT:
                    # Token sayısını yeniden tokenize etmiyoruz.
                    # Doğrudan orijinal token listesinin uzunluğunu
                    # kullanıyoruz.
                    "token_count": len(chunk_token_ids)
                }
            )

        # -------------------------------------------------
        # SON CHUNK
        # -------------------------------------------------

        if end >= len(token_ids):
            break

        start += step

    return chunks


# =========================================================
# ANA PROGRAM
# =========================================================

def main():

    # -----------------------------------------------------
    # OUTPUT KLASÖRÜ
    # -----------------------------------------------------

    CHUNKS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # TEMİZ TXT DOSYALARINI BUL
    # -----------------------------------------------------

    txt_files = sorted(
        CLEAN_DATA_DIR.glob("*.txt")
    )

    print("\n" + "=" * 70)

    print(
        f"{CHUNK_SIZE} TOKEN / "
        f"{CHUNK_OVERLAP} OVERLAP CHUNKING"
    )

    print("=" * 70)

    print(
        f"\nBulunan TXT sayısı: "
        f"{len(txt_files)}"
    )

    print(
        f"Tokenizer: {MODEL_NAME}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    # -----------------------------------------------------
    # TÜM CHUNK'LAR
    # -----------------------------------------------------

    all_chunks = []

    global_chunk_id = 0

    # =====================================================
    # BELGELERİ İŞLE
    # =====================================================

    for txt_path in txt_files:

        print("\n" + "-" * 70)

        print(
            f"İşleniyor: "
            f"{txt_path.name}"
        )

        # -------------------------------------------------
        # METNİ OKU
        # -------------------------------------------------

        text = txt_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():

            print(
                "UYARI: Belge boş, atlanıyor."
            )

            continue

        # -------------------------------------------------
        # CHUNK OLUŞTUR
        # -------------------------------------------------

        chunks = split_text_by_tokens(
            text=text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        print(
            f"Oluşturulan chunk sayısı: "
            f"{len(chunks)}"
        )

        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------

        for chunk_index, chunk_data in enumerate(
            chunks
        ):

            chunk_text = chunk_data["text"]

            token_count = chunk_data["token_count"]

            # -------------------------------------------------
            # GÜVENLİK KONTROLÜ
            # -------------------------------------------------

            if token_count > CHUNK_SIZE:

                raise ValueError(
                    "\nChunk token sınırını aştı!\n"
                    f"Kaynak     : {txt_path.name}\n"
                    f"Chunk      : {chunk_index}\n"
                    f"Token      : {token_count}\n"
                    f"Limit      : {CHUNK_SIZE}"
                )

            # -------------------------------------------------
            # CHUNK KAYDI
            # -------------------------------------------------

            chunk_record = {
                "chunk_id": global_chunk_id,
                "source": txt_path.name,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "text": chunk_text
            }

            all_chunks.append(
                chunk_record
            )

            global_chunk_id += 1

    # =====================================================
    # JSON OLARAK KAYDET
    # =====================================================

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

    # =====================================================
    # İSTATİSTİKLER
    # =====================================================

    token_counts = [
        chunk["token_count"]
        for chunk in all_chunks
    ]

    print("\n" + "=" * 70)

    print("CHUNKING TAMAMLANDI")

    print("=" * 70)

    print(
        f"\nToplam chunk sayısı: "
        f"{len(all_chunks)}"
    )

    if token_counts:

        print(
            f"Minimum token sayısı: "
            f"{min(token_counts)}"
        )

        print(
            f"Maksimum token sayısı: "
            f"{max(token_counts)}"
        )

        print(
            f"Ortalama token sayısı: "
            f"{sum(token_counts) / len(token_counts):.2f}"
        )

    else:

        print(
            "Token istatistiği hesaplanamadı."
        )

    print(
        f"\nÇıktı dosyası: "
        f"{OUTPUT_FILE}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()