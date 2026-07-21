from pathlib import Path

from pypdf import PdfReader


RAW_DATA_DIR = Path("data/raw")


pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

print("=" * 60)
print("PDF METİN ÇIKARMA TESTİ")
print("=" * 60)

print(f"\nBulunan PDF sayısı: {len(pdf_files)}")


for pdf_path in pdf_files:

    print("\n" + "-" * 60)
    print(f"Dosya: {pdf_path.name}")

    try:
        reader = PdfReader(pdf_path)

        print(f"Sayfa sayısı: {len(reader.pages)}")

        extracted_text = ""

        for page in reader.pages[:3]:
            text = page.extract_text()

            if text:
                extracted_text += text

        extracted_text = extracted_text.strip()

        if extracted_text:

            print("Durum: METİN ÇIKARILABİLDİ ✅")

            print("\nİlk 300 karakter:")
            print(extracted_text[:300])

        else:

            print("Durum: METİN ÇIKARILAMADI ❌")
            print(
                "PDF taranmış/görüntü tabanlı olabilir."
            )

    except Exception as error:

        print("Durum: HATA ❌")
        print(f"Hata: {error}")


print("\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)