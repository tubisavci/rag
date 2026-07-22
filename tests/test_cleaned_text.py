from pathlib import Path


CLEAN_DATA_DIR = Path("data/clean")


def main():

    txt_files = sorted(CLEAN_DATA_DIR.glob("*.txt"))

    print("=" * 60)
    print("TEMİZLENMİŞ METİN KALİTE KONTROLÜ")
    print("=" * 60)

    print(f"\nBulunan TXT sayısı: {len(txt_files)}")

    for txt_path in txt_files:

        text = txt_path.read_text(encoding="utf-8")

        print("\n" + "-" * 60)
        print(f"Dosya: {txt_path.name}")
        print(f"Karakter sayısı: {len(text)}")

        # Dosyanın ortasından örnek al.
        middle = len(text) // 2

        start = max(0, middle - 250)
        end = min(len(text), middle + 250)

        sample = text[start:end]

        print("\nMetnin ortasından 500 karakter:")
        print(sample)

    print("\n" + "=" * 60)
    print("KALİTE KONTROLÜ TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()