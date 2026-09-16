"""
RAG Prompt Template

Retrieval ve reranking sonucunda elde edilen
context bilgisini LLM'e gönderilecek prompt
formatına dönüştürür.
"""


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
Sen Türkçe çalışan bir soru-cevap asistanısın.

Görevin, sana verilen CONTEXT içerisindeki bilgilere
dayanarak kullanıcının sorusunu cevaplamaktır.

Kurallar:
1. Öncelikle ve yalnızca verilen CONTEXT bilgisini kullan.
2. CONTEXT içerisinde sorunun cevabı bulunmuyorsa bilgi uydurma.
3. Cevabı Türkçe üret.
4. Cevabı açık, anlaşılır ve mümkün olduğunca doğrudan ver.
5. CONTEXT içerisinde birden fazla ilgili bilgi varsa
   bunları anlamlı şekilde birleştir.
6. Sorunun cevabı CONTEXT içerisinde bulunmuyorsa:
   "Verilen kaynaklarda bu soruyu cevaplayacak yeterli
   bilgi bulunamadı."
   şeklinde belirt.
"""


# --------------------------------------------------
# USER PROMPT TEMPLATE
# --------------------------------------------------

USER_PROMPT_TEMPLATE = """
Aşağıda soruyu cevaplamak için kullanılabilecek
kaynak metinler verilmiştir.

CONTEXT:
--------------------
{context}
--------------------

SORU:
{question}

Yukarıdaki CONTEXT'e dayanarak soruyu cevapla.
"""


# --------------------------------------------------
# PROMPT OLUŞTURMA
# --------------------------------------------------

def build_prompt(context: str, question: str) -> str:
    """
    RAG için kullanıcı prompt'unu oluşturur.

    Args:
        context: Retrieval ve reranking sonucunda elde
                 edilen ilgili doküman parçaları.
        question: Kullanıcının sorusu.

    Returns:
        LLM'e gönderilecek prompt.
    """

    if not context or not context.strip():
        raise ValueError("Context boş olamaz.")

    if not question or not question.strip():
        raise ValueError("Soru boş olamaz.")

    return USER_PROMPT_TEMPLATE.format(
        context=context.strip(),
        question=question.strip(),
    )


# --------------------------------------------------
# CHAT MESAJLARI
# --------------------------------------------------

def build_messages(context: str, question: str):
    """
    Ollama gibi chat tabanlı LLM istemcileri için
    system ve user mesajlarını oluşturur.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": build_prompt(
                context,
                question,
            ),
        },
    ]


# --------------------------------------------------
# CONTEXT FORMATLAMA
# --------------------------------------------------

def format_context(results) -> str:
    """
    Hybrid Search + Reranking sonucundaki chunk'ları
    LLM'in kullanabileceği tek bir context metnine
    dönüştürür.

    Beklenen yapı:

        [
            (metadata, rerank_score),
            ...
        ]

    Metadata içerisinde 'text' alanı bulunmalıdır.
    """

    context_parts = []

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        text = chunk.get("text", "").strip()

        if not text:
            continue

        source = chunk.get(
            "source",
            "Bilinmeyen kaynak",
        )

        context_parts.append(
            f"[Kaynak {rank}: {source}]\n"
            f"{text}"
        )

    if not context_parts:
        raise ValueError(
            "Retrieval sonucunda kullanılabilir "
            "context bulunamadı."
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# CONTEXT LISTESİNDEN PROMPT OLUŞTURMA
# --------------------------------------------------

def create_rag_prompt(
    question: str,
    contexts: list[str],
) -> str:
    """
    Context metinlerinden RAG prompt'u oluşturur.

    Bu fonksiyon özellikle doğrudan metin listesi
    üzerinden prompt oluşturmak için kullanılır.
    """

    if not isinstance(question, str):
        raise TypeError("question string olmalıdır.")

    if not question.strip():
        raise ValueError("question boş olamaz.")

    if not isinstance(contexts, list):
        raise TypeError("contexts list olmalıdır.")

    if not contexts:
        raise ValueError("contexts boş olamaz.")

    valid_contexts = []

    for context in contexts:
        if not isinstance(context, str):
            raise TypeError(
                "contexts içerisindeki elemanlar string olmalıdır."
            )

        if context.strip():
            valid_contexts.append(context.strip())

    if not valid_contexts:
        raise ValueError(
            "contexts içerisinde kullanılabilir metin bulunamadı."
        )

    context_text = "\n\n".join(
        f"[Bağlam {i + 1}]\n{context}"
        for i, context in enumerate(valid_contexts)
    )

    return build_prompt(
        context=context_text,
        question=question,
    )


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_context = """
Türkiye'nin Ulusal Yapay Zeka Stratejisi,
yapay zeka alanında yetkin insan kaynağının
geliştirilmesini ve araştırma kapasitesinin
artırılmasını hedeflemektedir.
"""

    test_question = (
        "Ulusal Yapay Zeka Stratejisi'nin "
        "insan kaynağı açısından amacı nedir?"
    )

    prompt = build_prompt(
        test_context,
        test_question,
    )

    print("=" * 70)
    print("PROMPT TEMPLATE TESTİ")
    print("=" * 70)

    print("\nSYSTEM PROMPT:")
    print(SYSTEM_PROMPT)

    print("\nUSER PROMPT:")
    print(prompt)

    print("\n" + "=" * 70)
