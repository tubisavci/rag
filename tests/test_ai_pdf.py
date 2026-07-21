from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(
    "data/raw/yapayzeka_ulusal-yapay-zeka-stratejisi.pdf"
)

reader = PdfReader(PDF_PATH)

print(f"Toplam sayfa sayısı: {len(reader.pages)}")

test_pages = [0, 2, 5, 10, 20, 30, 50, 70, 90]

for page_number in test_pages:

    if page_number >= len(reader.pages):
        continue

    text = reader.pages[page_number].extract_text()

    print("\n" + "=" * 60)
    print(f"PDF sayfası: {page_number + 1}")
    print("=" * 60)

    if text and text.strip():

        cleaned_text = text.strip()

        print(
            f"Çıkarılan karakter sayısı: "
            f"{len(cleaned_text)}"
        )

        print("\nİlk 500 karakter:")
        print(cleaned_text[:500])

    else:

        print("Bu sayfadan metin çıkarılamadı.")