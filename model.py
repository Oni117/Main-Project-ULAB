"""
model.py
--------
Architecture matches the training script EXACTLY:
  - EfficientNet_Model class (features → pool → flatten → classifier)
  - IMAGE_SIZE = 240  (from training script)
  - Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  - Saved with: torch.save(model.state_dict(), "efficientnet_b1.pth")
"""

import io
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
# ===============================
# CBAM MODULE
# ===============================
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
# ── Must match your training IMAGE_SIZE ──────────────────────────
IMAGE_SIZE = 240

# ── Class order = alphabetical sort of your dataset folder names ──
# Run in Python to confirm:  sorted(os.listdir(r"D:\project\Mydata"))
# Adjust the list below to match that output exactly.
LABELS = ["Diabetic", "Non-Diabetic"]   # index-0, index-1


# ════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (copy-pasted from your training script)
# ════════════════════════════════════════════════════════════════
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        backbone = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.DEFAULT
        )

        self.features = backbone.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════
MODEL_FILE = r"C:\Users\sahal\OneDrive\Desktop\Main Project\efficientnet_b1.pth"


def load_model(model_name: str) -> nn.Module:
    model = EfficientNet_CBAM(num_classes=len(LABELS))

    if not os.path.exists(MODEL_FILE):
        print(
            f"[WARNING] '{MODEL_FILE}' not found.\n"
            "          Place your trained .pth file at backend/models/efficientnet_b1.pth\n"
            "          Running with RANDOM weights for now (results will be meaningless)."
        )
        model.eval()
        return model

    # Load state_dict (matches: torch.save(model.state_dict(), "efficientnet_b1.pth"))
    state_dict = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Model loaded from '{MODEL_FILE}'")
    return model


# ════════════════════════════════════════════════════════════════
# PREPROCESSING  (must match training transform exactly)
# ════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════
def predict_image(image_bytes: bytes, model: nn.Module) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)          # (1, 3, 240, 240)

    with torch.no_grad():
        logits = model(tensor)                    # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]  # (2,)
        pred   = torch.argmax(probs).item()

    return {
        "label":      LABELS[pred],
        "confidence": round(float(probs[pred]) * 100, 2),
        "probabilities": {
            LABELS[0]: round(float(probs[0]) * 100, 2),
            LABELS[1]: round(float(probs[1]) * 100, 2),
        },
    }

