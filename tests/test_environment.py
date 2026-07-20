import sys
import torch
import langchain
import sentence_transformers

print("=" * 50)
print("✅ Environment Test")
print("=" * 50)

print(f"Python Version : {sys.version}")
print(f"Torch Version  : {torch.__version__}")
print(f"LangChain      : {langchain.__version__}")

print("Sentence Transformers : OK")
print("🎉 Everything works correctly!")