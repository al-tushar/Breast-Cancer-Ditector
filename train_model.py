import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── 1. Load Images ───────────────────────────────────────────────
DATASET_PATH = "E:/AI/BreastCancerProject/dataset"
IMG_SIZE = 50
images = []
labels = []

print("Loading images...")
for label in ["0", "1"]:
    folder = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(folder)[:2000]:
        if fname.endswith(".png"):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(int(label))

images = np.array(images) / 255.0
labels = np.array(labels)
print(f"Loaded {len(images)} images")

# ── 2. Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert to PyTorch tensors (channels first: N, C, H, W)
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0,3,1,2)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).permute(0,3,1,2)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),
                          batch_size=32, shuffle=False)

# ── 3. Build CNN ─────────────────────────────────────────────────
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BreastCancerCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# ── 4. Train ─────────────────────────────────────────────────────
train_acc_history = []
val_acc_history   = []

print("\nTraining...")
for epoch in range(10):
    model.train()
    correct = 0
    total   = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        preds    = (output > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    train_acc = correct / total

    # Validation
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            preds  = (output > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
    val_acc = correct / total

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    print(f"Epoch {epoch+1}/10 — Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

# ── 5. Save Model ─────────────────────────────────────────────────
torch.save(model.state_dict(),
           "E:/AI/BreastCancerProject/breast_cancer_cnn.pth")
print("\nModel saved!")

# ── 6. Plot Training History ──────────────────────────────────────
plt.figure(figsize=(6,4))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history,   label="Val Accuracy")
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("E:/AI/BreastCancerProject/training_history.png", dpi=150)
plt.show()
print("Done!")