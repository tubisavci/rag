"""
Tam RAG Pipeline

Akış:

Soru
  ↓
Hybrid Search
  ↓
Reranking
  ↓
Context Formatting
  ↓
Prompt Template
  ↓
Qwen3 8B
  ↓
Cevap
"""

import argparse
import time

from  src.search_hybrid import (
    get_paths,
    load_index,
    load_metadata,
    load_model,
    build_bm25,
    hybrid_search,
)

from src.prompting.prompt_template import (
    format_context,
    build_messages,
)

from src.llm.ollama_client import (
    generate_response,
)


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

DEFAULT_STRATEGY = "500_100"
DEFAULT_TOP_K = 5


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tam RAG Pipeline"
    )

    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=[
            "300_50",
            "500_100",
            "800_150",
        ],
        help="Chunk stratejisi",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retrieval sonrası kullanılacak sonuç sayısı",
    )

    return parser.parse_args()


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

def answer_question(
    question,
    strategy=DEFAULT_STRATEGY,
    top_k=DEFAULT_TOP_K,
):
    """
    Verilen soruya tam RAG pipeline üzerinden
    cevap üretir.
    """

    total_start = time.perf_counter()

    # --------------------------------------------------
    # 1. INDEX VE METADATA
    # --------------------------------------------------

    print("\n[1/5] FAISS ve metadata yükleniyor...")

    index_path, metadata_path = get_paths(
        strategy
    )

    faiss_index = load_index(
        index_path
    )

    metadata = load_metadata(
        metadata_path
    )

    # --------------------------------------------------
    # 2. EMBEDDING MODELİ
    # --------------------------------------------------

    print("\n[2/5] Embedding modeli yükleniyor...")

    model, _ = load_model()

    # --------------------------------------------------
    # 3. BM25
    # --------------------------------------------------

    print("\n[3/5] BM25 hazırlanıyor...")

    bm25, _ = build_bm25(
        metadata
    )

    # --------------------------------------------------
    # 4. HYBRID SEARCH + RERANKING
    # --------------------------------------------------

    print("\n[4/5] Hybrid Search + Reranking...")

    retrieval_start = time.perf_counter()

    results = hybrid_search(
        model=model,
        faiss_index=faiss_index,
        bm25=bm25,
        metadata=metadata,
        question=question,
        top_k=max(top_k * 2, 10),
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not results:
        return {
            "answer": (
                "Verilen kaynaklarda bu soruyu "
                "cevaplayacak yeterli bilgi bulunamadı."
            ),
            "results": [],
            "retrieval_time": retrieval_time,
            "total_time": (
                time.perf_counter()
                - total_start
            ),
        }

    # --------------------------------------------------
    # 5. CONTEXT
    # --------------------------------------------------

    context = format_context(
        results[:top_k]
    )

    print(
        f"\nToplam context chunk sayısı: "
        f"{min(len(results), top_k)}"
    )

    # --------------------------------------------------
    # 6. PROMPT
    # --------------------------------------------------

    messages = build_messages(
        context=context,
        question=question,
    )

    # --------------------------------------------------
    # 7. LLM
    # --------------------------------------------------

    print("\n[5/5] Qwen3 8B cevap üretiyor...")

    llm_start = time.perf_counter()

    answer = generate_response(
        messages
    )

    llm_time = (
        time.perf_counter()
        - llm_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    return {
        "answer": answer,
        "results": results[:top_k],
        "context": context,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total_time": total_time,
    }


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

def print_answer(
    question,
    result,
):
    print("\n" + "=" * 70)
    print("RAG SONUCU")
    print("=" * 70)

    print(f"\nSoru:\n{question}")

    print("\nCevap:")
    print(result["answer"])

    print("\n" + "-" * 70)

    print(
        f"Retrieval + Reranking süresi: "
        f"{result['retrieval_time']:.4f} sn"
    )

    print(
        f"LLM cevap süresi: "
        f"{result['llm_time']:.4f} sn"
    )

    print(
        f"Toplam süre: "
        f"{result['total_time']:.4f} sn"
    )

    print("\n" + "=" * 70)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_arguments()

    print("=" * 70)
    print("TAM RAG PIPELINE")
    print("=" * 70)

    print(
        f"\nChunk stratejisi : {args.strategy}"
    )

    print(
        f"Top-K            : {args.top_k}"
    )

    while True:

        question = input(
            "\nSorunuzu girin "
            "(çıkmak için q):\n> "
        ).strip()

        if question.lower() == "q":
            print("\nProgram sonlandırıldı.")
            break

        if not question:
            print("\nBoş soru giremezsiniz.")
            continue

        result = answer_question(
            question=question,
            strategy=args.strategy,
            top_k=args.top_k,
        )

        print_answer(
            question,
            result,
        )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()