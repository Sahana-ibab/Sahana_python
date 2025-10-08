import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

class VariableDepthCNN(nn.Module):
    def __init__(self, num_conv_layers=2):
        super(VariableDepthCNN, self).__init__()

        layers = []
        in_channels = 1  

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(32))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(0.25))
            in_channels = 32

        self.conv = nn.Sequential(*layers)

        final_size = 28 // (2 ** num_conv_layers)
        final_size = max(final_size, 1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * final_size * final_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_model(model, trainloader, criterion, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_error = 1 - correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Loss: {running_loss / len(trainloader):.4f} | "
              f"Train Error: {train_error:.4f}")
    return train_error

train_errors = []
num_layers_list = [1, 2, 3]

for num_layers in num_layers_list:
    print(f"\nTraining CNN with {num_layers} convolutional layers...")
    model = VariableDepthCNN(num_conv_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_error = train_model(model, trainloader, criterion, optimizer, num_epochs=3)
    train_errors.append(train_error)

plt.figure(figsize=(7, 5))
plt.plot(num_layers_list, train_errors, marker='o', color='royalblue', linewidth=2)
plt.title("Training Error vs Number of CNN Layers (MNIST)")
plt.xlabel("Number of Convolutional Layers")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
