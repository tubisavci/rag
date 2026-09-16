from src.llm_ollama import OllamaLLM
from src.prompting.prompt_template import create_rag_prompt


def main():
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

    # 1. RAG prompt oluştur
    prompt = create_rag_prompt(
        question=question,
        contexts=contexts,
    )

    print("=" * 60)
    print("OLUŞTURULAN PROMPT")
    print("=" * 60)
    print(prompt)

    # 2. Ollama + Qwen3
    llm = OllamaLLM()

    print("\n" + "=" * 60)
    print("QWEN3 CEVABI")
    print("=" * 60)

    answer = llm.generate(prompt)

    print(answer)


if __name__ == "__main__":
    main()