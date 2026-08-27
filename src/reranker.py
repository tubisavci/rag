"""
BGE Reranker

Hybrid Search sonuçlarını yeniden sıralamak için kullanılır.
"""

import time
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-reranker-base"

LOCAL_MODEL_PATH = (
    r"C:\Users\User\.cache\huggingface\hub"
    r"\models--BAAI--bge-reranker-base"
    r"\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8

# Model başlangıçta yüklenmez.
_tokenizer = None
_model = None


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_reranker():
    """
    BGE reranker modelini gerektiğinde yükler.

    Model daha önce yüklenmişse tekrar yüklenmez.
    """

    global _tokenizer
    global _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    print("=" * 70)
    print("BGE RERANKER")
    print("=" * 70)

    print(f"\nModel : {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    print("\nModel yükleniyor...")

    start = time.perf_counter()

    _tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
)

    _model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
)

    _model.to(DEVICE)
    _model.eval()

    elapsed = time.perf_counter() - start

    print(
        f"Model hazır ({elapsed:.2f} sn)"
    )

    return _tokenizer, _model


# --------------------------------------------------
# PAIR OLUŞTURMA
# --------------------------------------------------

def create_pairs(
    question: str,
    documents: List[str],
) -> List[List[str]]:
    """
    Soru ve doküman çiftleri oluşturur.
    """

    return [
        [question, doc]
        for doc in documents
    ]


# --------------------------------------------------
# RERANK
# --------------------------------------------------

@torch.no_grad()
def rerank(
    question: str,
    documents: List[str],
) -> List[Tuple[int, float]]:
    """
    Question ile documentleri yeniden sıralar.

    Geri dönüş:
        [
            (document_index, score),
            ...
        ]
    """

    if len(documents) == 0:
        return []

    tokenizer, model = load_reranker()

    pairs = create_pairs(
        question,
        documents,
    )

    scores = []

    start = time.perf_counter()

    for i in range(
        0,
        len(pairs),
        BATCH_SIZE,
    ):
        batch = pairs[
            i:i + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(
            **encoded
        )

        batch_scores = (
            outputs.logits
            .view(-1)
            .float()
            .cpu()
            .tolist()
        )

        scores.extend(
            batch_scores
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        f"Reranking süresi: "
        f"{elapsed:.4f} sn"
    )

    return [
        (
            int(index),
            float(score),
        )
        for index, score in ranked
    ]


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("BGE RERANKER TESTİ")
    print("=" * 70)

    question = (
        "Türkiye'nin Ulusal Yapay Zeka "
        "Stratejisi'nin temel amaçları nelerdir?"
    )

    documents = [
        (
            "Türkiye'nin Ulusal Yapay Zeka Stratejisi, "
            "yapay zeka ekosisteminin geliştirilmesini "
            "hedeflemektedir."
        ),
        (
            "Türkiye'de enerji politikaları ve "
            "yenilenebilir enerji yatırımları."
        ),
        (
            "Yapay zeka alanında yetkin insan kaynağının "
            "geliştirilmesi hedeflenmektedir."
        ),
    ]

    results = rerank(
        question,
        documents,
    )

    print("\nSonuçlar:")

    for index, score in results:
        print(
            f"Index: {index} | "
            f"Score: {score:.4f}"
        )