"""
Türkçe RAG sistemi için prompt şablonları.
"""


SYSTEM_PROMPT = """
Sen Türkçe soru-cevap yapan bir yapay zeka asistanısın.

Sana verilen BAĞLAM içerisindeki bilgilere dayanarak
kullanıcının sorusunu cevapla.

Kurallar:
1. Öncelikle verilen bağlamı kullan.
2. Bağlamda bulunmayan bilgileri kesin bir gerçekmiş gibi sunma.
3. Cevap bağlamda bulunmuyorsa bunu açıkça belirt.
4. Cevabı Türkçe ver.
5. Gereksiz uzunlukta cevap verme.
6. Sorunun doğrudan cevabını önce ver.
"""


def create_rag_prompt(question: str, contexts: list[str]) -> str:
    """
    Soru ve retrieval sonuçlarından RAG prompt'u oluşturur.

    Args:
        question: Kullanıcının sorusu.
        contexts: Retrieval tarafından getirilen metinler.

    Returns:
        LLM'e gönderilecek prompt.
    """

    if not isinstance(question, str):
        raise TypeError("question string olmalıdır.")

    if not question.strip():
        raise ValueError("question boş olamaz.")

    if not isinstance(contexts, list):
        raise TypeError("contexts list olmalıdır.")

    if not contexts:
        raise ValueError("contexts boş olamaz.")

    context_text = "\n\n".join(
        f"[Bağlam {i + 1}]\n{context}"
        for i, context in enumerate(contexts)
    )

    prompt = f"""
{SYSTEM_PROMPT}

BAĞLAM:
{context_text}

SORU:
{question}

CEVAP:
"""

    return prompt.strip()


def main():
    """Prompt template testidir."""

    question = (
        "Türkiye'de iklim değişikliğinin "
        "tarım üzerindeki etkileri nelerdir?"
    )

    contexts = [
        (
            "İklim değişikliği tarımsal üretimde sıcaklık "
            "artışı ve aşırı hava olaylarının artmasına "
            "neden olabilmektedir."
        ),
        (
            "Kuraklık ve yağış rejimindeki değişiklikler "
            "tarımsal üretimi ve su kaynaklarını "
            "olumsuz etkileyebilmektedir."
        ),
    ]

    prompt = create_rag_prompt(question, contexts)

    print("=" * 60)
    print("RAG PROMPT TEST")
    print("=" * 60)
    print(prompt)


if __name__ == "__main__":
    main()