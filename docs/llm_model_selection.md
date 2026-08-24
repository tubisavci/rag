\# LLM Model Seçimi ve 26. Gün Çalışmaları



\## 1. Günün Amacı



26\. gün kapsamında RAG sisteminin cevap üretim aşamasında kullanılacak Büyük Dil Modeli (LLM) için Ollama kurulmuş ve uygun bir model belirlenmiştir.



\## 2. Ollama Kurulumu



LLM'lerin yerel ortamda çalıştırılabilmesi amacıyla Ollama kullanılmıştır.



Kurulum sonrasında Ollama sürümü:



```text

0.32.15

```



olarak belirlenmiştir.



\## 3. Sistem Özellikleri



Model seçiminde kullanılan sistem özellikleri:



\* RAM: yaklaşık 64 GB

\* GPU: NVIDIA GeForce RTX 3050 Laptop GPU

\* GPU belleği: yaklaşık 4 GB

\* İşletim sistemi: Windows



Sistemde ayrıca Intel Iris Xe Graphics bulunmaktadır.



\## 4. LLM Model Seçimi



RAG sistemi için \*\*Qwen3 8B\*\* modeli seçilmiştir.



Model Ollama üzerinden aşağıdaki komut kullanılarak kurulmuştur:



```powershell

ollama pull qwen3:8b

```



Modelin kurulumu başarıyla tamamlanmıştır.



\## 5. Model Seçim Gerekçesi



Qwen3 8B modeli aşağıdaki nedenlerle tercih edilmiştir:



\* Türkçe soru-cevap görevlerinde kullanılabilecek yeterli dil modelleme kapasitesine sahip olması.

\* 8B parametre boyutunun cevap kalitesi ile hesaplama maliyeti arasında dengeli bir seçenek sunması.

\* Sistemde bulunan yaklaşık 64 GB RAM sayesinde yerel olarak çalıştırılabilmesi.

\* NVIDIA RTX 3050 Laptop GPU'nun sınırlı VRAM kapasitesine rağmen Ollama ile CPU/RAM kaynaklarından da yararlanılarak çalıştırılabilmesi.

\* RAG sisteminden getirilecek bağlamı işleyerek bağlama dayalı cevaplar üretebilecek uygun bir model olması.



\## 6. Türkçe Model Testi



Model aşağıdaki komut ile çalıştırılmıştır:



```powershell

ollama run qwen3:8b

```



İlk testte modele şu soru yöneltilmiştir:



> Türkiye'nin Ulusal Yapay Zeka Stratejisi'nin temel amaçları nelerdir?



Model Türkçe ve yapılandırılmış bir cevap üretmiştir.



İkinci testte:



> RAG sistemlerinde hybrid search neden semantic search'ten daha başarılı olabilir?



sorusu yöneltilmiştir.



Model bu soruya da Türkçe cevap üretmiş ve hybrid search ile semantic search arasındaki farkları açıklamıştır.



\## 7. Test Değerlendirmesi



Yapılan testlerde Qwen3 8B'nin Türkçe sorulara cevap üretebildiği gözlemlenmiştir.



Bununla birlikte, model RAG bağlamı verilmeden çalıştırıldığında cevaplarını kendi önceden öğrenilmiş bilgisine dayanarak üretmektedir. Bu nedenle cevapların belirli bir kaynak dokümandaki bilgiye dayandığı garanti edilememektedir.



Bu durum, geliştirilen RAG sisteminde retrieval aşamasından elde edilen kaynakların LLM'e context olarak aktarılmasının önemini göstermektedir.



İlerleyen aşamalarda aynı sorular RAG sistemi üzerinden test edilerek RAG kullanılmadığında ve kullanıldığında oluşan cevaplar karşılaştırılacaktır.



\## 8. 26. Gün Sonucu



26\. gün kapsamında:



\* Ollama kurulmuştur.

\* Ollama'nın çalıştığı doğrulanmıştır.

\* Sistem kaynakları incelenmiştir.

\* Qwen3 8B modeli seçilmiştir.

\* Qwen3 8B modeli Ollama üzerinden kurulmuştur.

\* Türkçe soru-cevap testleri gerçekleştirilmiştir.

\* Modelin RAG pipeline'ında kullanılmasına karar verilmiştir.



\*\*26. Gün: Tamamlandı.\*\*



