import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np


# =========================================================
# PROJE KÖK DİZİNİ
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from embedding_model import BGEEmbeddingModel


# =========================================================
# AYARLAR
# =========================================================

CHUNKS_DIR = BASE_DIR / "data" / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
RESULTS_DIR = BASE_DIR / "results"


STRATEGIES = {
    "300_50": {
        "chunk_file": CHUNKS_DIR / "chunks_300_50.json",
        "embedding_file": EMBEDDINGS_DIR / "embeddings_300_50.npy",
    },
    "500_100": {
        "chunk_file": CHUNKS_DIR / "chunks_500_100.json",
        "embedding_file": EMBEDDINGS_DIR / "embeddings_500_100.npy",
    },
    "800_150": {
        "chunk_file": CHUNKS_DIR / "chunks_800_150.json",
        "embedding_file": EMBEDDINGS_DIR / "embeddings_800_150.npy",
    },
}


# =========================================================
# TEST SORULARI
# =========================================================

QUESTIONS = [
    {
        "question": "Türkiye'nin çevre sorunları nelerdir?",
        "expected": "cevre_bakanlik",
    },
    {
        "question": "Milli Eğitim Bakanlığının 2024 faaliyetleri nelerdir?",
        "expected": "egitim_meb",
    },
    {
        "question": "Enerji verimliliği neden önemlidir?",
        "expected": "enerji_etkb",
    },
    {
        "question": "Gıda okuryazarlığı nedir?",
        "expected": "gida_tarimorman",
    },
    {
        "question": "Siber güvenlik nedir?",
        "expected": "siber_guvenlik",
    },
    {
        "question": "İklim değişikliğinin tarıma etkileri nelerdir?",
        "expected": "tarim_bakanlik",
    },
    {
        "question": "Tüketici haklarının amacı nedir?",
        "expected": "tuketici_ticaretbakanligi",
    },
    {
        "question": "Türkiye'nin uzay çalışmaları hangi kurum tarafından yürütülmektedir?",
        "expected": "uzay_tubitak",
    },
    {
        "question": "Ulusal Yapay Zeka Stratejisinin amacı nedir?",
        "expected": "yapayzeka",
    },
    {
        "question": "Türk Dil Kurumunun faaliyetleri nelerdir?",
        "expected": "dil_tdk",
    },
]


TOP_K = 5


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def load_chunks(path):
    """Chunk JSON dosyasını yükler."""

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_source_name(source):
    """
    Dosya adından kaynak kimliğini üretir.

    Örneğin:
    cevre_bakanlik_cevre-sorunlari-ve-oncelikleri.txt
    ->
    cevre_bakanlik
    """

    return source.split("_", 2)[0] + "_" + source.split("_", 2)[1]


def is_correct(source, expected):
    """Retrieved kaynağın beklenen belge olup olmadığını kontrol eder."""

    return expected in source


def calculate_mrr(rank):
    """
    Reciprocal Rank hesaplar.

    rank:
        1 tabanlı sıralama.
    """

    if rank is None:
        return 0.0

    return 1.0 / rank


def build_faiss_index(embeddings):
    """Embedding matrisinden FAISS index oluşturur."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# STRATEJİ TESTİ
# =========================================================

def test_strategy(
    strategy_name,
    strategy_info,
    model,
):
    """Tek bir chunk stratejisini test eder."""

    chunk_path = strategy_info["chunk_file"]
    embedding_path = strategy_info["embedding_file"]

    print("\n" + "=" * 70)
    print(f"CHUNK STRATEJİSİ: {strategy_name}")
    print("=" * 70)

    # -----------------------------------------------------
    # DOSYA KONTROLLERİ
    # -----------------------------------------------------

    if not chunk_path.exists():

        raise FileNotFoundError(
            f"Chunk dosyası bulunamadı: {chunk_path}"
        )

    if not embedding_path.exists():

        raise FileNotFoundError(
            f"Embedding dosyası bulunamadı: {embedding_path}"
        )

    # -----------------------------------------------------
    # CHUNKLARI YÜKLE
    # -----------------------------------------------------

    print(f"\nChunk dosyası : {chunk_path.name}")

    chunks = load_chunks(chunk_path)

    print(f"Chunk sayısı  : {len(chunks)}")

    # -----------------------------------------------------
    # EMBEDDINGLERİ YÜKLE
    # -----------------------------------------------------

    print(f"Embedding     : {embedding_path.name}")

    embeddings = np.load(
        embedding_path
    ).astype("float32")

    print(
        f"Embedding shape : {embeddings.shape}"
    )

    if len(chunks) != len(embeddings):

        raise ValueError(
            "Chunk sayısı ile embedding sayısı eşleşmiyor!"
        )

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    print("\nFAISS index oluşturuluyor...")

    index_start = time.perf_counter()

    index = build_faiss_index(
        embeddings
    )

    index_time = time.perf_counter() - index_start

    print(
        f"FAISS hazır ({index_time:.4f} sn)"
    )

    # -----------------------------------------------------
    # TEST SONUÇLARI
    # -----------------------------------------------------

    top1_correct = 0
    recall5_total = 0
    mrr5_total = 0

    retrieval_times = []

    detailed_results = []

    # -----------------------------------------------------
    # SORULAR
    # -----------------------------------------------------

    for question_index, item in enumerate(
        QUESTIONS,
        start=1
    ):

        question = item["question"]
        expected = item["expected"]

        print("\n" + "-" * 70)

        print(
            f"{question_index}/{len(QUESTIONS)}"
        )

        print(
            f"Soru: {question}"
        )

        print(
            f"Beklenen: {expected}"
        )

        # -------------------------------------------------
        # QUERY EMBEDDING + SEARCH
        # -------------------------------------------------

        start = time.perf_counter()

        query_embedding = model.encode(
            [question]
        )

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        elapsed = time.perf_counter() - start

        retrieval_times.append(
            elapsed
        )

        # -------------------------------------------------
        # SONUÇLAR
        # -------------------------------------------------

        retrieved_sources = []

        for rank, index_value in enumerate(
            indices[0],
            start=1
        ):

            if index_value < 0:
                continue

            source = chunks[index_value]["source"]

            retrieved_sources.append(
                source
            )

        # -------------------------------------------------
        # TOP-1
        # -------------------------------------------------

        top1_source = (
            retrieved_sources[0]
            if retrieved_sources
            else ""
        )

        top1_is_correct = is_correct(
            top1_source,
            expected
        )

        if top1_is_correct:
            top1_correct += 1

        # -------------------------------------------------
        # RECALL@5
        # -------------------------------------------------

        recall5_is_correct = any(
            is_correct(
                source,
                expected
            )
            for source in retrieved_sources
        )

        if recall5_is_correct:
            recall5_total += 1

        # -------------------------------------------------
        # MRR@5
        # -------------------------------------------------

        reciprocal_rank = 0.0
        correct_rank = None

        for rank, source in enumerate(
            retrieved_sources,
            start=1
        ):

            if is_correct(
                source,
                expected
            ):

                correct_rank = rank

                reciprocal_rank = calculate_mrr(
                    rank
                )

                break

        mrr5_total += reciprocal_rank

        # -------------------------------------------------
        # ÇIKTI
        # -------------------------------------------------

        print(
            "Top-1:",
            "EVET" if top1_is_correct else "HAYIR"
        )

        print(
            "Recall@5:",
            "EVET" if recall5_is_correct else "HAYIR"
        )

        if correct_rank is not None:

            print(
                f"İlk doğru sonuç sırası: {correct_rank}"
            )

        else:

            print(
                "İlk doğru sonuç sırası: BULUNAMADI"
            )

        print(
            f"Süre: {elapsed:.4f} sn"
        )

        # -------------------------------------------------
        # DETAYLI SONUÇ
        # -------------------------------------------------

        detailed_results.append(
            {
                "question": question,
                "expected": expected,
                "top1_source": top1_source,
                "top1_correct": top1_is_correct,
                "recall_at_5": recall5_is_correct,
                "correct_rank": correct_rank,
                "mrr_at_5": reciprocal_rank,
                "retrieval_time": elapsed,
            }
        )

    # =====================================================
    # İSTATİSTİKLER
    # =====================================================

    question_count = len(QUESTIONS)

    top1_accuracy = (
        top1_correct / question_count
    )

    recall5 = (
        recall5_total / question_count
    )

    mrr5 = (
        mrr5_total / question_count
    )

    average_time = (
        sum(retrieval_times)
        / len(retrieval_times)
    )

    # =====================================================
    # SONUÇ
    # =====================================================

    print("\n" + "=" * 70)
    print(
        f"SONUÇ: {strategy_name}"
    )
    print("=" * 70)

    print(
        f"\nToplam chunk      : {len(chunks)}"
    )

    print(
        f"Top-1 Accuracy    : {top1_accuracy:.4f}"
    )

    print(
        f"Recall@5          : {recall5:.4f}"
    )

    print(
        f"MRR@5             : {mrr5:.4f}"
    )

    print(
        f"Ortalama süre     : {average_time:.4f} sn"
    )

    return {
        "strategy": strategy_name,
        "chunk_count": len(chunks),
        "top1_accuracy": top1_accuracy,
        "recall_at_5": recall5,
        "mrr_at_5": mrr5,
        "average_retrieval_time": average_time,
        "index_build_time": index_time,
        "details": detailed_results,
    }


# =========================================================
# ANA PROGRAM
# =========================================================

def main():

    print("=" * 70)
    print("RAG CHUNK STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 70)

    print(
        "\nStratejiler:"
    )

    for strategy in STRATEGIES:

        print(
            f"  - {strategy}"
        )

    print(
        f"\nTest soru sayısı: {len(QUESTIONS)}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\nEmbedding modeli yükleniyor...")

    model = BGEEmbeddingModel()

    print("\nModel hazır.")

    # -----------------------------------------------------
    # TÜM STRATEJİLER
    # -----------------------------------------------------

    results = []

    for strategy_name, strategy_info in STRATEGIES.items():

        result = test_strategy(
            strategy_name,
            strategy_info,
            model,
        )

        results.append(
            result
        )

    # =====================================================
    # KARŞILAŞTIRMA TABLOSU
    # =====================================================

    print("\n\n" + "=" * 80)
    print("CHUNK STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 80)

    print(
        "\n"
        f"{'Strateji':<15}"
        f"{'Chunk':<10}"
        f"{'Top-1':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR@5':<12}"
        f"{'Süre':<12}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['strategy']:<15}"
            f"{result['chunk_count']:<10}"
            f"{result['top1_accuracy'] * 100:>7.2f}%   "
            f"{result['recall_at_5'] * 100:>7.2f}%   "
            f"{result['mrr_at_5']:.4f}       "
            f"{result['average_retrieval_time']:.4f}s"
        )

    # =====================================================
    # EN İYİ STRATEJİLER
    # =====================================================

    best_accuracy = max(
        results,
        key=lambda x: x["top1_accuracy"]
    )

    best_mrr = max(
        results,
        key=lambda x: x["mrr_at_5"]
    )

    fastest = min(
        results,
        key=lambda x: x["average_retrieval_time"]
    )

    print("\n" + "=" * 80)
    print("ÖNE ÇIKAN SONUÇLAR")
    print("=" * 80)

    print(
        f"\nEn yüksek Top-1 Accuracy : "
        f"{best_accuracy['strategy']} "
        f"({best_accuracy['top1_accuracy'] * 100:.2f}%)"
    )

    print(
        f"En yüksek MRR@5          : "
        f"{best_mrr['strategy']} "
        f"({best_mrr['mrr_at_5']:.4f})"
    )

    print(
        f"En hızlı retrieval       : "
        f"{fastest['strategy']} "
        f"({fastest['average_retrieval_time']:.4f} sn)"
    )

    # =====================================================
    # JSON KAYDET
    # =====================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / "chunk_strategy_comparison.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nSonuçlar kaydedildi:"
    )

    print(
        output_path
    )

    print("\n" + "=" * 80)
    print("CHUNK KARŞILAŞTIRMASI TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()
