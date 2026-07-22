from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Yapay zekâ, bilgisayar sistemlerinin insan zekâsı gerektiren görevleri
gerçekleştirebilmesini sağlayan teknolojiler bütünüdür. Günümüzde yapay zekâ
sağlık, eğitim, ulaşım ve finans gibi birçok alanda kullanılmaktadır.

Doğal dil işleme, yapay zekânın önemli çalışma alanlarından biridir.
Bilgisayarların insan dilini anlaması, yorumlaması ve üretmesi üzerine çalışır.
Soru-cevap sistemleri de doğal dil işleme uygulamalarından biridir.

Retrieval Augmented Generation sistemleri, büyük dil modellerini dış bilgi
kaynaklarıyla destekler. Kullanıcının sorusuyla ilgili belgeler önce bilgi
tabanından bulunur ve daha sonra dil modeline bağlam olarak gönderilir.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


chunks = splitter.split_text(text)


print("=" * 60)
print("RECURSIVE CHARACTER TEXT SPLITTER TESTİ")
print("=" * 60)

print(f"\nToplam chunk sayısı: {len(chunks)}")


for index, chunk in enumerate(chunks, start=1):

    print("\n" + "-" * 60)
    print(f"CHUNK {index}")
    print(f"Karakter sayısı: {len(chunk)}")
    print("-" * 60)

    print(chunk)


print("\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)