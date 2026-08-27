"""
BGE-M3 Embedding Model

BGE-M3 modelini Transformers kullanarak yükler
ve GPU üzerinde embedding üretir.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# --------------------------------------------------
# MODEL AYARLARI
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"

LOCAL_MODEL_PATH = (
    r"C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-m3"
    r"\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8
MAX_LENGTH = 512


class BGEEmbeddingModel:
    """BGE-M3 embedding modeli."""

    def __init__(
        self,
        model_name=MODEL_NAME,
        device=DEVICE,
    ):
        self.model_name = model_name
        self.device = device

        print("\nBGE-M3 yükleniyor...")
        print(f"Device: {self.device}")
        print(f"Model path: {LOCAL_MODEL_PATH}")

        start = time.perf_counter()

        # --------------------------------------------------
        # YEREL MODEL
        # --------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
        )

        self.model.to(device)
        self.model.eval()

        elapsed = time.perf_counter() - start

        print(
            f"BGE-M3 hazır "
            f"({elapsed:.2f} sn)"
        )

    @torch.no_grad()
    def encode(
        self,
        texts,
        batch_size=BATCH_SIZE,
    ):
        """Metinleri BGE-M3 embeddinglerine dönüştürür."""

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                i:i + batch_size
            ]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0]

            # Normalize
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

        embeddings = np.vstack(
            all_embeddings
        ).astype("float32")

        return embeddings


def main():
    """Embedding modeli test."""

    print("=" * 70)
    print("BGE-M3 EMBEDDING MODEL TESTİ")
    print("=" * 70)

    model = BGEEmbeddingModel()

    texts = [
        "Türkiye'nin yapay zekâ stratejisi.",
        "Yapay zekâ alanında insan kaynağının geliştirilmesi.",
    ]

    embeddings = model.encode(texts)

    print("\n" + "=" * 70)
    print("TEST SONUCU")
    print("=" * 70)

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Embedding dtype:",
        embeddings.dtype,
    )

    print(
        "Device:",
        model.device,
    )

    if torch.cuda.is_available():
        print(
            "VRAM:",
            round(
                torch.cuda.memory_allocated()
                / 1024**2,
                2,
            ),
            "MB",
        )

    print("\nTEST BAŞARILI")


if __name__ == "__main__":
    main()