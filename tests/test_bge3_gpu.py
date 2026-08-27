import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "BAAI/bge-m3"

print("=" * 70)
print("BGE-M3 GPU EMBEDDING TESTİ")
print("=" * 70)

print(f"\nCUDA: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA kullanılamıyor!")

print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nModel yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

model = model.to("cuda")
model.eval()

print("Model cihazı:", next(model.parameters()).device)

text = (
    "Türkiye'nin Ulusal Yapay Zeka Stratejisi "
    "yapay zeka ekosistemini geliştirmeyi hedeflemektedir."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

inputs = {
    key: value.to("cuda")
    for key, value in inputs.items()
}

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    outputs = model(**inputs)

embedding = F.normalize(
    outputs.last_hidden_state[:, 0],
    p=2,
    dim=1,
)

print("\n" + "=" * 70)
print("EMBEDDING SONUCU")
print("=" * 70)

print("Embedding OK")
print("Boyut:", embedding.shape)
print("Cihaz:", embedding.device)

print(
    "Anlık VRAM:",
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2,
    ),
    "MB",
)

print(
    "Peak VRAM:",
    round(
        torch.cuda.max_memory_allocated() / 1024**2,
        2,
    ),
    "MB",
)

print("=" * 70)
print("TEST BAŞARILI")
print("=" * 70)