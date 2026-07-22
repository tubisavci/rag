import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


# ---------------------------------------------------------
# KLASÖR AYARLARI
# ---------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
CLEAN_DATA_DIR = Path("data/clean")


# ---------------------------------------------------------
# PDF METİN ÇIKARMA
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """
    PDF içerisindeki metni sayfa sayfa çıkarır.

    Tamamen boş olan veya anlamlı metin içermeyen
    sayfalar sonuçlara eklenmez.
    """

    reader = PdfReader(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        # "a", "bc" gibi aşırı kısa sayfaları alma.
        if len(text) < 20:
            continue

        extracted_pages.append(text)

    return "\n\n".join(extracted_pages)


# ---------------------------------------------------------
# METİN TEMİZLEME
# ---------------------------------------------------------

def clean_text(text):
    """
    PDF'den çıkarılan metindeki temel biçim ve
    karakter bozukluklarını temizler.
    """

    # Unicode karakterlerini standartlaştır.
    text = unicodedata.normalize("NFKC", text)

    # -----------------------------------------------------
    # PDF Unicode kalıntıları
    # Örnek:
    # /uni015E -> Ş
    # /uni015F -> ş
    # -----------------------------------------------------

    unicode_replacements = {
        "/uni015E": "Ş",
        "/uni015F": "ş",
        "/uni011E": "Ğ",
        "/uni011F": "ğ",
        "/uni0130": "İ",
        "/uni0131": "ı",
        "/uni00C7": "Ç",
        "/uni00E7": "ç",
        "/uni00D6": "Ö",
        "/uni00F6": "ö",
        "/uni00DC": "Ü",
        "/uni00FC": "ü",
    }

    for broken, correct in unicode_replacements.items():
        text = text.replace(broken, correct)

    # -----------------------------------------------------
    # Kontrol karakterlerini temizle
    # -----------------------------------------------------

    text = text.replace("\x00", " ")

    # Tab karakterlerini boşluğa çevir.
    text = text.replace("\t", " ")

    # -----------------------------------------------------
    # Satır içindeki gereksiz boşlukları temizle
    # -----------------------------------------------------

    text = re.sub(r"[ ]{2,}", " ", text)

    # Satır sonunda kalan boşlukları temizle.
    text = re.sub(r"[ \t]+\n", "\n", text)

    # -----------------------------------------------------
    # Gereksiz fazla satır boşluklarını azalt
    # -----------------------------------------------------

    text = re.sub(r"\n{3,}", "\n\n", text)

    # -----------------------------------------------------
    # Satır sonu tire bölünmelerini birleştir
    #
    # Örnek:
    # sürdürüle-
    # bilir
    #
    # ->
    #
    # sürdürülebilir
    # -----------------------------------------------------

    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text
    )

    # -----------------------------------------------------
    # Tek satır kırılmalarını boşluğa dönüştür
    #
    # Paragrafları belirten çift satır kırılmaları korunur.
    # -----------------------------------------------------

    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text
    )

    # Tekrar oluşabilecek fazla boşlukları düzelt.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Paragrafların başındaki ve sonundaki boşlukları temizle.
    paragraphs = []

    for paragraph in text.split("\n\n"):

        paragraph = paragraph.strip()

        if paragraph:
            paragraphs.append(paragraph)

    text = "\n\n".join(paragraphs)

    return text.strip()


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

def main():

    CLEAN_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        RAW_DATA_DIR.glob("*.pdf")
    )

    print("=" * 60)
    print("PDF METİN ÇIKARMA VE TEMİZLEME")
    print("=" * 60)

    print(
        f"\nBulunan PDF sayısı: "
        f"{len(pdf_files)}"
    )

    for pdf_path in pdf_files:

        print("\n" + "-" * 60)

        print(
            f"İşleniyor: "
            f"{pdf_path.name}"
        )

        # PDF metnini çıkar.
        raw_text = extract_text_from_pdf(
            pdf_path
        )

        # Temizle.
        cleaned_text = clean_text(
            raw_text
        )

        print(
            f"Ham karakter sayısı: "
            f"{len(raw_text)}"
        )

        print(
            f"Temiz karakter sayısı: "
            f"{len(cleaned_text)}"
        )

        output_path = (
            CLEAN_DATA_DIR
            / f"{pdf_path.stem}.txt"
        )

        output_path.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(
            f"Kaydedildi: "
            f"{output_path}"
        )

    print("\n" + "=" * 60)

    print(
        "METİN ÇIKARMA VE TEMİZLEME "
        "TAMAMLANDI"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()