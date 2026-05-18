from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # 👉 On charge maintenant le modèle CNN-only
    model_path = os.environ.get("MODEL_PATH", "..\\models\\hybrid_autofeat.h5")

    if not os.path.exists(model_path):
        print(f"⚠️  Modèle non trouvé à '{model_path}'. Mode démonstration activé.")
    else:
        try:
            import tensorflow as tf
            # 👉 Aucun custom object, aucun compile, simple et propre
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Modèle CNN-only chargé depuis '{model_path}'")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")

    yield
    model = None


app = FastAPI(title="CIFAR-10 Animal Classifier", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

ANIMAL_CLASSES = {
    "oiseau": {"emoji": "🐦", "en": "bird"},
    "chat": {"emoji": "🐱", "en": "cat"},
    "cerf": {"emoji": "🦌", "en": "deer"},
    "chien": {"emoji": "🐶", "en": "dog"},
    "grenouille": {"emoji": "🐸", "en": "frog"},
    "cheval": {"emoji": "🐴", "en": "horse"},
    "avion": {"emoji": "✈️", "en": "airplane"},
    "automobile": {"emoji": "🚗", "en": "automobile"},
    "bateau": {"emoji": "🚢", "en": "ship"},
    "camion": {"emoji": "🚛", "en": "truck"},
}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image to CIFAR-10 format (32x32 RGB)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32), Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def demo_prediction(image_bytes: bytes) -> dict:
    """Fake prediction for demo mode (no model loaded)."""
    import hashlib
    h = int(hashlib.md5(image_bytes[:100]).hexdigest(), 16)
    idx = h % 10
    
    probs = np.random.dirichlet(np.ones(10) * 0.3)
    probs[idx] = max(probs[idx], 0.55)
    probs = probs / probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    
    return {
        "predicted_class": CIFAR10_CLASSES[idx],
        "confidence": float(probs[idx]),
        "emoji": ANIMAL_CLASSES[CIFAR10_CLASSES[idx]]["emoji"],
        "is_animal": CIFAR10_CLASSES[idx] in ["oiseau", "chat", "cerf", "chien", "grenouille", "cheval"],
        "top5": [
            {
                "class": CIFAR10_CLASSES[i],
                "confidence": float(probs[i]),
                "emoji": ANIMAL_CLASSES[CIFAR10_CLASSES[i]]["emoji"]
            }
            for i in sorted_idx[:5]
        ],
        "demo_mode": True
    }


@app.get("/")
async def root():
    return {"message": "CIFAR-10 Classifier API", "status": "running", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")
    
    contents = await file.read()
    
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image trop grande (max 10MB).")
    
    if model is None:
        return JSONResponse(content=demo_prediction(contents))
    
    try:
        img_array = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image: {str(e)}")
    
    try:
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_class = CIFAR10_CLASSES[predicted_idx]
        sorted_idx = np.argsort(predictions)[::-1]
        
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "emoji": ANIMAL_CLASSES[predicted_class]["emoji"],
            "is_animal": predicted_class in ["oiseau", "chat", "cerf", "chien", "grenouille", "cheval"],
            "top5": [
                {
                    "class": CIFAR10_CLASSES[i],
                    "confidence": float(predictions[i]),
                    "emoji": ANIMAL_CLASSES[CIFAR10_CLASSES[i]]["emoji"]
                }
                for i in sorted_idx[:5]
            ],
            "demo_mode": False
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
