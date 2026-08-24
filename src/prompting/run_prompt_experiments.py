"""
29. Gün - Gerçek Prompt Engineering Deneyi

Aynı RAG context'i ve aynı soru ile
farklı prompt stratejilerini Qwen3 8B
üzerinde karşılaştırır.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import time
import json

from search_hybrid import (
    get_paths,
    load_index,
    load_metadata,
    load_model,
    build_bm25,
    hybrid_search,
)

from prompt_experiments import (
    build_experiment_prompt,
)

from llm.ollama_client import (
    generate_response,
)


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

STRATEGY = "500_100"
TOP_K = 5

QUESTION = (
    "Türkiye'nin Ulusal Yapay Zeka Stratejisi'nin "
    "temel amaçları nelerdir?"
)


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def get_context():

    print("=" * 70)
    print("PROMPT ENGINEERING GERÇEK DENEY")
    print("=" * 70)

    print("\n[1] FAISS ve metadata yükleniyor...")

    index_path, metadata_path = get_paths(
        STRATEGY
    )

    faiss_index = load_index(
        index_path
    )

    metadata = load_metadata(
        metadata_path
    )

    print("\n[2] BGE-M3 yükleniyor...")

    model, _ = load_model()

    print("\n[3] BM25 hazırlanıyor...")

    bm25, _ = build_bm25(
        metadata
    )

    print("\n[4] Hybrid Search + Reranking...")

    results = hybrid_search(
        model=model,
        faiss_index=faiss_index,
        bm25=bm25,
        metadata=metadata,
        question=QUESTION,
        top_k=max(TOP_K * 2, 10),
    )

    results = results[:TOP_K]

    context_parts = []

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):

        text = chunk.get(
            "text",
            "",
        ).strip()

        source = chunk.get(
            "source",
            "Bilinmeyen kaynak",
        )

        context_parts.append(
            f"[Kaynak {rank}: {source}]\n"
            f"{text}"
        )

    return "\n\n".join(
        context_parts
    )


# --------------------------------------------------
# DENEY
# --------------------------------------------------

def save_results(results):
    """
    Prompt deney sonuçlarını JSON dosyasına kaydeder.
    """

    output_path = Path(
        "results/prompt_experiments.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\nDeney sonuçları kaydedildi: "
        f"{output_path}"
    )

def run_experiment(
    strategy,
    context,
):

    print("\n" + "=" * 70)
    print(
        f"PROMPT STRATEJİSİ: "
        f"{strategy.upper()}"
    )
    print("=" * 70)

    messages = build_experiment_prompt(
        strategy=strategy,
        context=context,
        question=QUESTION,
    )

    print("\nQwen3 8B cevap üretiyor...")

    start = time.perf_counter()

    try:
    answer = generate_response(
        messages
    )

except Exception as error:
    print("\nDENEY HATASI:")
    print(error)

    return {
        "strategy": strategy,
        "answer": None,
        "llm_time": None,
        "error": str(error),
    }

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\nCEVAP:")
    print(answer)

    print(
        f"\nLLM süresi: "
        f"{elapsed:.4f} saniye"
    )

    return {
        "strategy": strategy,
        "answer": answer,
        "llm_time": elapsed,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    context = get_context()

    print("\n" + "=" * 70)
    print("KULLANILAN SORU")
    print("=" * 70)

    print(QUESTION)

    print("\n" + "=" * 70)
    print("DENEYLER BAŞLIYOR")
    print("=" * 70)

    results = []

    for strategy in [
        "simple",
        "rag",
        "source",
    ]:

        result = run_experiment(
            strategy,
            context,
        )

        results.append(
            result
        )

    print("\n" + "=" * 70)
    print("DENEY TAMAMLANDI")
    print("=" * 70)

    for result in results:

    print(
        f"\n{result['strategy'].upper()}"
    )

    if result["llm_time"] is not None:

        print(
            f"LLM süresi: "
            f"{result['llm_time']:.4f} sn"
        )

    else:

        print("Durum: HATA")

        print(
            f"Hata: "
            f"{result.get('error', 'Bilinmeyen hata')}"
        )

save_results(results)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()