"""
Ollama LLM Client

Qwen3 8B modeline Ollama üzerinden
istek göndermek için kullanılır.
"""

import requests


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:8b"


# --------------------------------------------------
# LLM ÇAĞRISI
# --------------------------------------------------

def generate_response(
    messages,
    temperature=0.2,
):
    """
    Ollama üzerinden Qwen3 8B modeline
    mesaj gönderir.

    Args:
        messages: System ve user mesajlarını içeren liste.
        temperature: Modelin cevap üretim sıcaklığı.

    Returns:
        LLM tarafından üretilen cevap.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_messages = [
        {
            "role": "system",
            "content": (
                "Sen Türkçe cevap veren "
                "bir asistansın."
            ),
        },
        {
            "role": "user",
            "content": (
                "RAG nedir? Kısaca açıkla."
            ),
        },
    ]

    print("=" * 70)
    print("OLLAMA CLIENT TESTİ")
    print("=" * 70)

    print(f"\nModel: {MODEL_NAME}")

    print("\nModel cevap üretiyor...")

    answer = generate_response(
        test_messages
    )

    print("\nCEVAP:")
    print(answer)

    print("\n" + "=" * 70)
    print("TEST BAŞARILI")
    print("=" * 70)