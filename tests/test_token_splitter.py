from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


print("Tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer başarıyla yüklendi.")


def token_length(text):
    """
    Bir metnin tokenizer'a göre kaç token olduğunu döndürür.
    """
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)


test_text = """
Retrieval Augmented Generation sistemleri, büyük dil modellerinin dış bilgi
kaynaklarından yararlanmasını sağlayan bir yaklaşımdır. Kullanıcının sorgusu
öncelikle uygun bir temsil biçimine dönüştürülür ve bilgi tabanındaki ilgili
dokümanlar aranır.

Bilgi erişim aşamasında sorguyla anlamsal olarak ilişkili metin parçaları
belirlenir. Elde edilen parçalar büyük dil modeline bağlam olarak gönderilir.
Böylece model yalnızca önceden öğrendiği bilgilerle değil, sisteme sağlanan
güncel veya alan özelindeki bilgilerle de cevap oluşturabilir.
"""


chunks = splitter.split_text(test_text)


print("\n" + "=" * 60)
print("TOKEN BAZLI CHUNKING TESTİ")
print("=" * 60)

print(f"\nToplam chunk sayısı: {len(chunks)}")


for index, chunk in enumerate(chunks, start=1):

    print("\n" + "-" * 60)
    print(f"CHUNK {index}")
    print(f"Token sayısı: {token_length(chunk)}")
    print(f"Karakter sayısı: {len(chunk)}")
    print("-" * 60)

    print(chunk)


print("\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)