import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
import itertools

# Loading and transposing
landmark_df = pd.read_csv("landmark_genes.csv", index_col=0).T
landmark_df = landmark_df.iloc[1:, :]
target_df = pd.read_csv("target_genes.csv", index_col=0).T
target_df = target_df.iloc[1:, :]

landmark_df = landmark_df.apply(pd.to_numeric, errors="coerce")
target_df   = target_df.apply(pd.to_numeric, errors="coerce")

# print("NaNs after conversion:", landmark_df.isna().sum().sum())
# Fill NaNs with column mean
landmark_df = landmark_df.fillna(landmark_df.mean())
target_df   = target_df.fillna(target_df.mean())

# Splitting it to: train 80%, temp 20%
lm_train, lm_temp, tg_train, tg_temp = train_test_split(
    landmark_df, target_df, test_size=0.2, random_state=42
)

# Splitting it to: temp into val 10%, test 10%
lm_val, lm_test, tg_val, tg_test = train_test_split(
    lm_temp, tg_temp, test_size=0.5, random_state=42
)

# --------------------------------MODEL:----------------------------------------------

# Z-Score Normalization of sample:

mean = lm_train.mean(axis=0)
std = lm_train.std(axis=0)

lm_train = (lm_train - mean)/std
lm_test = (lm_test - mean)/std
lm_val = (lm_val - mean)/std

y_mean = tg_train.mean(axis=0)
y_std = tg_train.std(axis=0)

tg_train = (tg_train - y_mean)/y_std
tg_val   = (tg_val   - y_mean)/y_std
tg_test  = (tg_test  - y_mean)/y_std

X_train = torch.tensor(lm_train.values, dtype=torch.float32)
y_train = torch.tensor(tg_train.values, dtype=torch.float32)

X_test = torch.tensor(lm_test.values, dtype=torch.float32)
y_test = torch.tensor(tg_test.values, dtype=torch.float32)

X_val = torch.tensor(lm_val.values, dtype=torch.float32)
y_val = torch.tensor(tg_val.values, dtype=torch.float32)

# Dataset loading:-----------------

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)

#  Defining FFN:-------------------

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=0.3):
        super(FFN, self).__init__()
        self.lay1 = nn.Linear(input_dim, hidden_dim)
        self.batnorm = nn.BatchNorm1d(hidden_dim)
        self.lay2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.batnorm2 = nn.BatchNorm1d(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)

        self.lay3 = nn.Linear(hidden_dim*2, output_dim)


    def forward(self, x):
        x = self.lay1(x)
        x = self.batnorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lay2(x)
        x = self.batnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lay3(x)

        return x

def train_and_validate(model, train_loader, val_loader, lr=0.001, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_loader.dataset)

    return val_loss

def hyperparam_search(X_train, y_train, X_val, y_val, input_dim, output_dim):
    param_grid = {
        "hidden_dim": [32, 64, 128],
        "dropout": [0.2, 0.3, 0.5],
        "lr": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64]
    }

    best_val_loss = float("inf")
    best_params = None
    best_state = None

    for hidden_dim, dropout, lr, batch_size in itertools.product(
            param_grid["hidden_dim"],
            param_grid["dropout"],
            param_grid["lr"],
            param_grid["batch_size"]
    ):
        print(f"\n Training with hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, batch_size={batch_size}")

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = FFN(input_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)

        val_loss = train_and_validate(model, train_loader, val_loader, lr=lr, epochs=20)
        print(f"Validation Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (hidden_dim, dropout, lr, batch_size)
            best_state = model.state_dict()

    print("\nBest Hyperparameters:", best_params, "with Val Loss:", best_val_loss)
    return best_params, best_state

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
print(output_dim)
# Hyperparameter search
best_params, best_state = hyperparam_search(X_train, y_train, X_val, y_val, input_dim, output_dim)

hidden_dim, dropout, lr, batch_size = best_params

trainval_ds = TensorDataset(torch.cat([X_train, X_val]), torch.cat([y_train, y_val]))
trainval_loader = DataLoader(trainval_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

final_model = FFN(input_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)

optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Retraining the model on train and val set with best parameters
for epoch in range(50):
    final_model.train()
    train_loss = 0.0
    for Xb, yb in trainval_loader:
        optimizer.zero_grad()
        pred = final_model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(trainval_loader.dataset)

    # Optional: evaluate on val each epoch if you want
    print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}")

# Test evaluation
final_model.eval()
test_loss = 0.0
with torch.no_grad():
    for Xb, yb in test_loader:
        pred = final_model(Xb)
        loss = loss_fn(pred, yb)
        test_loss += loss.item() * Xb.size(0)
test_loss /= len(test_loader.dataset)

print(f"\nFinal Test Loss = {test_loss :.4f}")











