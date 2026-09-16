"""
Chunk-Level Relevance Review

Aday chunk'ları insan değerlendirmesine sunar.

Relevance:
0 = İlgisiz
1 = Kısmen ilgili
2 = Doğrudan ilgili
"""

import json
from pathlib import Path


# ==========================================================
# PATH
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "chunk_relevance_candidates.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "chunk_relevance_ground_truth.json"
)


# ==========================================================
# DOSYA KONTROLÜ
# ==========================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Aday dosyası bulunamadı: {INPUT_PATH}"
    )


# ==========================================================
# JSON OKU
# ==========================================================

with INPUT_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    questions = json.load(file)


if not isinstance(questions, list):
    raise ValueError(
        "JSON list formatında olmalıdır."
    )


# ==========================================================
# ESKİ DEĞERLENDİRMELERİ YÜKLE
# ==========================================================

if OUTPUT_PATH.exists():

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        reviewed_questions = json.load(file)

else:

    reviewed_questions = questions


# ==========================================================
# BAŞLIK
# ==========================================================

print("=" * 80)
print("CHUNK-LEVEL RELEVANCE DEĞERLENDİRMESİ")
print("=" * 80)

print(
    "\nRelevance değerleri:"
)

print(
    "0 = İlgisiz"
)

print(
    "1 = Kısmen ilgili"
)

print(
    "2 = Doğrudan ilgili"
)

print(
    "q = Çıkış"
)


# ==========================================================
# SORULAR
# ==========================================================

for question in reviewed_questions:

    question_id = question["question_id"]

    question_text = question["question"]

    candidates = question["candidates"]


    print("\n" + "=" * 80)

    print(
        f"SORU {question_id}"
    )

    print("=" * 80)

    print(
        f"\nSoru:\n{question_text}"
    )

    print(
        f"\nBeklenen kaynak:\n"
        f"{question['expected_source']}"
    )

    print(
        f"\nAday chunk sayısı: "
        f"{len(candidates)}"
    )


    # ------------------------------------------------------
    # CHUNKLAR
    # ------------------------------------------------------

    for candidate_number, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            "\n" + "-" * 80
        )

        print(
            f"ADAY {candidate_number}/{len(candidates)}"
        )

        print(
            "-" * 80
        )

        print(
            f"Chunk ID      : "
            f"{candidate['chunk_id']}"
        )

        print(
            f"Chunk Index   : "
            f"{candidate['chunk_index']}"
        )

        print(
            f"Kaynak        : "
            f"{candidate['source']}"
        )

        print(
            f"Token Sayısı  : "
            f"{candidate['token_count']}"
        )

        print(
            "\nBulunduğu yöntemler:"
        )

        found_by = candidate["found_by"]

        methods = []

        if found_by["semantic"]:
            methods.append("Semantic")

        if found_by["bm25"]:
            methods.append("BM25")

        if found_by["hybrid"]:
            methods.append("Hybrid")

        if found_by["reranker"]:
            methods.append("Reranker")

        print(
            ", ".join(methods)
            if methods
            else "-"
        )

        print(
            "\nCHUNK METNİ:"
        )

        print("-" * 80)

        print(
            candidate["text"]
        )

        print("-" * 80)


        # --------------------------------------------------
        # MEVCUT DEĞER
        # --------------------------------------------------

        current = candidate.get(
            "relevant"
        )

        print(
            f"\nMevcut relevance: "
            f"{current}"
        )

        while True:

            value = input(
                "Relevance [0/1/2, Enter=0, q=çıkış]: "
            ).strip().lower()

            if value == "q":

                # O ana kadarki sonuçları kaydet
                with OUTPUT_PATH.open(
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        reviewed_questions,
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    "\nDeğerlendirme kaydedildi."
                )

                print(
                    OUTPUT_PATH
                )

                raise SystemExit


            if value == "":
                value = "0"


            if value in {
                "0",
                "1",
                "2",
            }:

                candidate["relevant"] = int(
                    value
                )

                break


            print(
                "Hatalı giriş. "
                "Sadece 0, 1, 2 veya q girin."
            )


# ==========================================================
# SONUÇLARI KAYDET
# ==========================================================

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        reviewed_questions,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ==========================================================
# ÖZET
# ==========================================================

counts = {
    0: 0,
    1: 0,
    2: 0,
}

for question in reviewed_questions:

    for candidate in question["candidates"]:

        value = candidate.get(
            "relevant"
        )

        if value in counts:
            counts[value] += 1


print(
    "\n" + "=" * 80
)

print(
    "DEĞERLENDİRME TAMAMLANDI"
)

print(
    "=" * 80
)

print(
    f"İlgisiz (0)       : {counts[0]}"
)

print(
    f"Kısmen ilgili (1) : {counts[1]}"
)

print(
    f"Doğrudan ilgili (2): {counts[2]}"
)

print(
    f"\nÇıktı:"
)

print(
    OUTPUT_PATH
)