import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

print("Tokenizer loaded. Loading model...")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

print("Model loaded into memory")

# Just to check it's usable
sample = tokenizer("ATGCGTATGC", return_tensors="pt")
with torch.no_grad():
    outputs = model(**sample)
print("Model ran on a dummy input")
