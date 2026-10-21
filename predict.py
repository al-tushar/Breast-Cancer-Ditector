import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog

# ── Same CNN architecture (must match train_model.py) ─────────────
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

# ── Load trained model ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = BreastCancerCNN().to(device)
model.load_state_dict(torch.load(
    "E:/AI/BreastCancerProject/breast_cancer_cnn.pth",
    map_location=device))
model.eval()
print("Model loaded successfully!")

# ── Open file picker window ───────────────────────────────────────
root = tk.Tk()
root.withdraw()
IMAGE_PATH = filedialog.askopenfilename(
    title="Select a Breast Cancer Image",
    initialdir="E:/AI/BreastCancerProject/",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
)

if not IMAGE_PATH:
    print("No image selected. Exiting.")
    exit()

print(f"Selected: {IMAGE_PATH}")

# ── Process & Predict ─────────────────────────────────────────────
img         = Image.open(IMAGE_PATH).convert("RGB")
img_resized = img.resize((50, 50))
img_array   = np.array(img_resized) / 255.0
img_tensor  = torch.tensor(img_array, dtype=torch.float32)
img_tensor  = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(img_tensor).item()

label      = "MALIGNANT" if prediction > 0.5 else "BENIGN"
confidence = prediction if prediction > 0.5 else 1 - prediction
color      = "red" if label == "MALIGNANT" else "green"

# ── Show Result ───────────────────────────────────────────────────
plt.figure(figsize=(5, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction : {label}\nConfidence : {confidence*100:.2f}%",
          fontsize=14, color=color, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("E:/AI/BreastCancerProject/prediction_result.png", dpi=150)
plt.show()

print(f"\n{'='*40}")
print(f"  File      : {IMAGE_PATH.split('/')[-1]}")
print(f"  Result    : {label}")
print(f"  Confidence: {confidence*100:.2f}%")
print(f"{'='*40}")