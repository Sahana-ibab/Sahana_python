# feature_extractor.py
import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


resnet = models.resnet50(pretrained=True)

resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)


def extract_features(image_dir, output_file="image_features.pkl"):
    features = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(image_files, desc="Extracting features"):
        img_path = os.path.join(image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = resnet(image)
                feat = feat.view(-1).cpu().numpy()

            features[img_name] = feat
        except Exception as e:
            print(f" Error with {img_name}: {e}")

    # Save features as pickle
    with open(output_file, "wb") as f:
        pickle.dump(features, f)

    print(f" Saved {len(features)} image features {output_file}")


if __name__ == "__main__":
    image_dir = "Images"
    extract_features(image_dir, "image_features.pkl")
