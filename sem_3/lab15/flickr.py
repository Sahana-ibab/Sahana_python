import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

with open("captions.pkl", "rb") as f:
    captions_data = pickle.load(f)

word2idx = captions_data["word2idx"]
idx2word = captions_data["idx2word"]
sequences = captions_data["captions_indices"]  # dict: img_name -> list of captions (indices)
vocab_size = len(word2idx)

sequences = {k.split(",")[0]: v for k, v in sequences.items()}

with open("image_features.pkl", "rb") as f:
    image_features = pickle.load(f)


all_features = []
all_captions = []

for img, caps in sequences.items():
    if img not in image_features:
        continue  # skip images without features
    for cap in caps:
        all_features.append(torch.tensor(image_features[img], dtype=torch.float32))
        all_captions.append(torch.tensor(cap, dtype=torch.long))

print(f"Total training examples: {len(all_features)}")

class CaptionDataset(Dataset):
    def __init__(self, features, captions):
        self.features = features
        self.captions = captions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.captions[idx]

# Collate function for padding
def collate_fn(batch):
    feats, caps = zip(*batch)
    feats = torch.stack(feats, dim=0)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    return feats, caps

dataset = CaptionDataset(all_features, all_captions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# RNN
class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc_feat = nn.Linear(2048, embed_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        features = self.fc_feat(features).unsqueeze(1)  # (batch, 1, embed_size)
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.rnn(inputs)
        outputs = self.fc_out(hiddens)
        return outputs

    def generate(self, feature, max_len=20):
        result = []
        with torch.no_grad():
            feature = self.fc_feat(feature).unsqueeze(0).unsqueeze(1)
            states = None
            input = feature
            for _ in range(max_len):
                hiddens, states = self.rnn(input, states)
                output = self.fc_out(hiddens.squeeze(1))
                predicted = output.argmax(1).item()
                word = idx2word[predicted]
                if word == "<EOS>":
                    break
                if word == "<end>":
                    break
                result.append(word)
                input = self.embed(torch.tensor([predicted])).unsqueeze(1)
        return " ".join(result)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptionGenerator(embed_size=256, hidden_size=512, vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for feats, caps in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        feats, caps = feats.to(device), caps.to(device)
        outputs = model(feats, caps)
        loss = criterion(outputs.reshape(-1, vocab_size), caps.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Test
test_img = list(image_features.keys())[0]
test_feat = torch.tensor(image_features[test_img], dtype=torch.float32).to(device)
caption = model.generate(test_feat, max_len=15)
print(f"Image: {test_img}")
print(f"Generated Caption: {caption}")
