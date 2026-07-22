from pathlib import Path
from contextlib import asynccontextmanager
import base64
import json
import io

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model import load_model

# ===============================
# PATH SETUP
# ===============================
BASE_DIR         = Path(__file__).resolve().parent
MODEL_PATH       = BASE_DIR / "models" / "efficientnet_b1+cbam_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# ===============================
# DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# LOAD CLASS NAMES
# ===============================
if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

# Maps raw folder names → display names shown in the UI
DISPLAY_NAMES = {
    "diabetes":     "Diabetic",
    "nondiabetes":  "Non-Diabetic",
}

def display(label: str) -> str:
    return DISPLAY_NAMES.get(label.lower(), label.capitalize())

# ===============================
# IMAGE TRANSFORM
# ===============================
IMAGE_SIZE = 240

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

# ===============================
# FASTAPI LIFESPAN
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        device=device,
    )
    print("Model loaded successfully.")
    yield
    print("Shutting down...")

# ===============================
# APP INSTANCE
# ===============================
app = FastAPI(title="TongueCheck API", version="3.0.0", lifespan=lifespan)

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_gradcam(model, input_tensor, target_class):
    activations = []
    gradients  = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer   = model.features[-1]
    fwd_handle     = target_layer.register_forward_hook(fwd_hook)
    bwd_handle     = target_layer.register_full_backward_hook(bwd_hook)

    try:
        output = model(input_tensor)
        loss   = output[:, target_class]
        model.zero_grad()
        loss.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM could not capture activations/gradients.")

        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * activations[0]).sum(dim=1))
        cam     = cam.squeeze().detach().cpu().numpy()
        cam     = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam    -= cam.min()
        cam    /= cam.max() + 1e-8
        return cam

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def build_gradcam_images(raw_img: Image.Image, cam: np.ndarray):
    img_np  = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    overlay = np.clip(heatmap * 0.45 + img_np * 0.55, 0, 255).astype(np.uint8)
    return {
        "original": pil_to_data_url(Image.fromarray(img_np)),
        "heatmap":  pil_to_data_url(Image.fromarray(heatmap)),
        "overlay":  pil_to_data_url(Image.fromarray(overlay)),
    }


# ===============================
# ROUTES
# ===============================
@app.get("/")
def root():
    return {
        "message":     "TongueCheck API is running",
        "device":      str(device),
        "num_classes": NUM_CLASSES,
        "classes":     CLASS_NAMES,
    }


@app.get("/health")
def health_check():
    return {
        "status":       "healthy",
        "model_loaded": True,
        "device":       str(device),
        "classes":      CLASS_NAMES,
    }


@app.post("/predict")
async def predict(
    file:       UploadFile = File(...),
    model_name: str        = Query(default="efficientnet_b1_cbam"),
):
    try:
        image_bytes      = await file.read()
        image            = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor     = transform(image).unsqueeze(0).to(device)

        model = app.state.model
        model.eval()

        with torch.no_grad():
            outputs          = model(input_tensor)
            probabilities    = torch.softmax(outputs, dim=1)[0]
            predicted_index  = int(torch.argmax(probabilities).item())
            confidence       = round(float(probabilities[predicted_index].item()) * 100, 2)

        raw_label = CLASS_NAMES[predicted_index]

        return {
            "label":      display(raw_label),
            "confidence": confidence,
            "model_used": model_name,
            "probabilities": {
                display(CLASS_NAMES[i]): round(float(probabilities[i].item()) * 100, 2)
                for i in range(NUM_CLASSES)
            },
        }

    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gradcam")
async def gradcam(
    file:       UploadFile = File(...),
    model_name: str        = Query(default="efficientnet_b1_cbam"),
):
    try:
        image_bytes  = await file.read()
        raw_img      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        model = app.state.model
        model.eval()

        output           = model(input_tensor)
        probabilities    = torch.softmax(output, dim=1)[0]
        pred_index       = int(torch.argmax(probabilities).item())
        confidence       = round(float(probabilities[pred_index].item()) * 100, 2)

        cam    = generate_gradcam(model, input_tensor, pred_index)
        images = build_gradcam_images(raw_img, cam)

        return {
            "pred_class": display(CLASS_NAMES[pred_index]),
            "confidence": confidence,
            "model_used": model_name,
            **images,
        }

    except Exception as e:
        print("Grad-CAM error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# STATIC FILES — must be LAST
# Serves the built React frontend
# ===============================
app.mount("/", StaticFiles(directory="static", html=True), name="static")
