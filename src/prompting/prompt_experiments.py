"""
29. Gün - Prompt Engineering Deneyleri

Farklı prompt stratejilerini karşılaştırmak
amacıyla kullanılır.
"""


# --------------------------------------------------
# PROMPT 1 - BASİT
# --------------------------------------------------

SIMPLE_SYSTEM_PROMPT = """
Sen Türkçe cevap veren bir asistansın.
"""


# --------------------------------------------------
# PROMPT 2 - RAG ODAKLI
# --------------------------------------------------

RAG_SYSTEM_PROMPT = """
Sen Türkçe çalışan bir soru-cevap asistanısın.

Kurallar:
1. Yalnızca verilen CONTEXT bilgisini kullan.
2. CONTEXT içerisinde cevap yoksa bilgi uydurma.
3. Cevabı Türkçe ver.
4. Cevabı açık ve anlaşılır şekilde oluştur.
5. Gereksiz bilgi ekleme.
"""


# --------------------------------------------------
# PROMPT 3 - KAYNAK ODAKLI
# --------------------------------------------------

SOURCE_AWARE_SYSTEM_PROMPT = """
Sen Türkçe çalışan bir soru-cevap asistanısın.

Kurallar:
1. Yalnızca verilen CONTEXT bilgisini kullan.
2. CONTEXT içerisinde cevap yoksa bilgi uydurma.
3. Cevabı Türkçe ver.
4. Cevabı açık ve anlaşılır şekilde oluştur.
5. Cevabı destekleyen kaynakları belirt.
6. CONTEXT'te bulunmayan bilgileri kendi bilginden ekleme.
"""


# --------------------------------------------------
# USER PROMPT
# --------------------------------------------------

USER_PROMPT_TEMPLATE = """
CONTEXT:
--------------------
{context}
--------------------

SORU:
{question}

Cevabı verilen CONTEXT'e dayanarak oluştur.
"""


# --------------------------------------------------
# PROMPT OLUŞTURMA
# --------------------------------------------------

def build_experiment_prompt(
    strategy,
    context,
    question,
):
    """
    Seçilen prompt stratejisine göre
    system ve user mesajlarını oluşturur.
    """

    prompts = {
        "simple": SIMPLE_SYSTEM_PROMPT,
        "rag": RAG_SYSTEM_PROMPT,
        "source": SOURCE_AWARE_SYSTEM_PROMPT,
    }

    if strategy not in prompts:
        raise ValueError(
            f"Geçersiz prompt stratejisi: {strategy}"
        )

    return [
        {
            "role": "system",
            "content": prompts[strategy].strip(),
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context.strip(),
                question=question.strip(),
            ),
        },
    ]


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_context = """
Türkiye'nin Ulusal Yapay Zeka Stratejisi,
yapay zeka alanında yetkin insan kaynağının
geliştirilmesini hedeflemektedir.
"""

    test_question = (
        "Ulusal Yapay Zeka Stratejisi'nin "
        "insan kaynağı açısından amacı nedir?"
    )

    strategies = [
        "simple",
        "rag",
        "source",
    ]

    print("=" * 70)
    print("PROMPT ENGINEERING DENEYLERİ")
    print("=" * 70)

    for strategy in strategies:

        print("\n" + "=" * 70)
        print(f"STRATEJİ: {strategy.upper()}")
        print("=" * 70)

        messages = build_experiment_prompt(
            strategy=strategy,
            context=test_context,
            question=test_question,
        )

        print("\nSYSTEM:")
        print(messages[0]["content"])

        print("\nUSER:")
        print(messages[1]["content"])

    print("\n" + "=" * 70)
    print("TEST BAŞARILI")
    print("=" * 70)