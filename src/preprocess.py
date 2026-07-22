import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


# ---------------------------------------------------------
# KLASÖR AYARLARI
# ---------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
CLEAN_DATA_DIR = Path("data/clean")

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "preprocessing.log"

def setup_logging():
    """
    Preprocessing işlemlerini hem terminale
    hem de logs/preprocessing.log dosyasına kaydeder.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                LOG_FILE,
                mode="w",
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ],
        force=True
    )


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

    # Log sistemini başlat.
    setup_logging()

    # Clean klasörü yoksa oluştur.
    CLEAN_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Raw klasöründeki PDF dosyalarını bul.
    pdf_files = sorted(
        RAW_DATA_DIR.glob("*.pdf")
    )

    total_count = len(pdf_files)
    success_count = 0
    error_count = 0

    logging.info("=" * 60)
    logging.info("PDF METİN ÇIKARMA VE TEMİZLEME BAŞLATILDI")
    logging.info("=" * 60)

    logging.info(
        "Toplam %d PDF bulundu.",
        total_count
    )

    # Hiç PDF bulunamazsa uyar.
    if total_count == 0:
        logging.warning(
            "data/raw klasöründe PDF bulunamadı."
        )
        return

    for pdf_path in pdf_files:

        logging.info("-" * 60)

        logging.info(
            "İşleniyor: %s",
            pdf_path.name
        )

        try:
            # PDF metnini çıkar.
            raw_text = extract_text_from_pdf(
                pdf_path
            )

            # Metni temizle.
            cleaned_text = clean_text(
                raw_text
            )

            # Temiz metin tamamen boşsa hata kabul et.
            if not cleaned_text:
                raise ValueError(
                    "Temizleme sonucunda metin boş kaldı."
                )

            # Çıktı dosyasının yolunu oluştur.
            output_path = (
                CLEAN_DATA_DIR
                / f"{pdf_path.stem}.txt"
            )

            # Temiz metni UTF-8 olarak kaydet.
            output_path.write_text(
                cleaned_text,
                encoding="utf-8"
            )

            success_count += 1

            logging.info(
                "BAŞARILI: %s",
                pdf_path.name
            )

            logging.info(
                "Ham karakter sayısı: %d",
                len(raw_text)
            )

            logging.info(
                "Temiz karakter sayısı: %d",
                len(cleaned_text)
            )

            logging.info(
                "Kaydedildi: %s",
                output_path
            )

        except Exception as error:

            error_count += 1

            logging.exception(
                "HATA: %s | %s",
                pdf_path.name,
                error
            )

    logging.info("=" * 60)
    logging.info("PREPROCESSING ÖZETİ")
    logging.info("=" * 60)

    logging.info(
        "Toplam belge : %d",
        total_count
    )

    logging.info(
        "Başarılı     : %d",
        success_count
    )

    logging.info(
        "Hatalı       : %d",
        error_count
    )

    logging.info("=" * 60)

    if error_count == 0:
        logging.info(
            "Tüm belgeler başarıyla işlendi."
        )
    else:
        logging.warning(
            "%d belge işlenirken hata oluştu.",
            error_count
        )


if __name__ == "__main__":
    main()