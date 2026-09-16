"""
Ollama tabanlı LLM modülü.

Bu modül, yerel Ollama sunucusu üzerinden Qwen3 modeline
prompt göndererek cevap üretir.
"""

import requests


class OllamaLLM:
    """Ollama API üzerinden LLM cevapları üretir."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Verilen prompt için Ollama üzerinden cevap üretir.

        Args:
            prompt: LLM'e gönderilecek prompt.

        Returns:
            Model tarafından üretilen cevap.
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt string olmalıdır.")

        if not prompt.strip():
            raise ValueError("prompt boş olamaz.")

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if "response" not in data:
            raise RuntimeError(
                "Ollama API cevabında 'response' alanı bulunamadı."
            )

        return data["response"].strip()


def main():
    """Basit Ollama bağlantı testi."""

    llm = OllamaLLM()

    prompt = (
        "Türkiye'nin başkenti neresidir? "
        "Türkçe ve kısa cevap ver."
    )

    answer = llm.generate(prompt)

    print("=" * 60)
    print("OLLAMA + QWEN3 TEST")
    print("=" * 60)
    print(f"Model: {llm.model_name}")
    print(f"Cevap: {answer}")


if __name__ == "__main__":
    main()