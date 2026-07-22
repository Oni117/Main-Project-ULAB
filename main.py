from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn, os

from model import load_model, predict_image

# ── Loaded models live here ──
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    ml_models["efficientnet_b1"] = load_model("efficientnet_b1")
    print("Model ready.")
    yield
    ml_models.clear()

app = FastAPI(title="TongueCheck API", version="1.0.0", lifespan=lifespan)

# ── CORS: allow React dev server ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def root():
    return {"status": "ok", "message": "TongueCheck API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(ml_models.keys()),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG, PNG or WEBP."
        )

    image_bytes = await file.read()

    # Validate size
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    if "efficientnet_b1" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    try:
        result = predict_image(image_bytes, ml_models["efficientnet_b1"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

