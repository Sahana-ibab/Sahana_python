import pickle
import re

#  Loading captions
print("Loading captions...")
with open("captions.txt", "r", encoding="utf-8") as f:
    captions = f.read().strip().split("\n")

print(f"Loaded {len(captions)} captions")

# Preprocessing text text
def tokenize(sentence):
    sentence = sentence.lower()
    tokens = re.findall(r"\b\w+\b", sentence)
    return tokens

# Splitting into (image, caption) pairs
image_caption_pairs = []
for line in captions:
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        continue
    img, cap = parts
    image_caption_pairs.append((img, tokenize(cap)))

print("Example tokenization:")
print(f"Original: {captions[0]}")
print(f"Tokens: {image_caption_pairs[0][1]}")

print("Building vocabulary...")

# Special tokens
special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# Counting words
word_freq = {}
for _, cap in image_caption_pairs:
    for word in cap:
        word_freq[word] = word_freq.get(word, 0) + 1

# Creating final vocabulary
vocab = special_tokens + sorted(word_freq.keys())
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
print(f"Vocabulary size: {len(vocab)} words")

print("Converting captions to sequences...")

captions_indices = {}
for img, cap in image_caption_pairs:
    seq = [word2idx["<SOS>"]]
    for w in cap:
        seq.append(word2idx.get(w, word2idx["<UNK>"]))
    seq.append(word2idx["<EOS>"])
    if img not in captions_indices:
        captions_indices[img] = []
    captions_indices[img].append(seq)

print("Example sequence:")
print("   Image:", list(captions_indices.keys())[0])
print("   Indices:", captions_indices[list(captions_indices.keys())[0]][0])

# Saving as pickle file:
data = {
    "captions_indices": captions_indices, 
    "word2idx": word2idx,
    "idx2word": idx2word
}

with open("captions.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved as captions.pkl")
