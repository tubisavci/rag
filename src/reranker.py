"""
BGE Reranker

Hybrid Search sonuçlarını yeniden sıralamak için kullanılır.
"""

import time
from typing import List

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-reranker-base"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8

# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("=" * 70)
print("BGE RERANKER")
print("=" * 70)

print(f"\nModel : {MODEL_NAME}")
print(f"Device: {DEVICE}")

print("\nModel yükleniyor...")

start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

elapsed = time.perf_counter() - start

print(
    f"Model hazır ({elapsed:.2f} sn)"
)

# --------------------------------------------------
# PAIR OLUŞTUR
# --------------------------------------------------

def create_pairs(
    question: str,
    documents: List[str],
):
    """
    Question-document çiftleri oluşturur.
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
):
    """
    Question ile documentleri yeniden sıralar.

    Geri dönüş:
        [
            (index, score),
            ...
        ]
    """

    if len(documents) == 0:
        return []

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

        batch_pairs = pairs[
            i:i + BATCH_SIZE
        ]

        inputs = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        logits = model(
            **inputs
        ).logits

        batch_scores = (
            logits.squeeze(-1)
            .cpu()
            .tolist()
        )

        if isinstance(
            batch_scores,
            float,
        ):
            batch_scores = [
                batch_scores
            ]

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
        f"\nReranking süresi: "
        f"{elapsed:.4f} sn"
    )

    return ranked

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("RERANKER TESTİ")
    print("=" * 70)

    question = "Türkiye'nin çevre sorunları nelerdir?"

    documents = [

        "Türkiye'de hava kirliliği, su kirliliği ve atık yönetimi önemli çevre sorunlarıdır.",

        "Python programlama dili Guido van Rossum tarafından geliştirilmiştir.",

        "Çevre, Şehircilik ve İklim Değişikliği Bakanlığı çevre politikalarını yürütmektedir.",

        "Türkiye'nin uzay çalışmaları TÜBİTAK UZAY tarafından desteklenmektedir.",

        "İklim değişikliği tarımı olumsuz etkilemektedir."

    ]

    print(f"\nSoru:\n{question}")

    print("\nDokümanlar:")

    for i, doc in enumerate(documents, start=1):

        print(f"{i}. {doc}")

    ranked = rerank(
        question,
        documents,
    )

    print("\n" + "=" * 70)
    print("RERANK SONUCU")
    print("=" * 70)

    for rank, (doc_index, score) in enumerate(
        ranked,
        start=1,
    ):

        print(f"\n{rank}. SONUÇ")

        print(
            f"Skor : {score:.4f}"
        )

        print(
            f"Doküman : {documents[doc_index]}"
        )

    print("\n" + "=" * 70)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()