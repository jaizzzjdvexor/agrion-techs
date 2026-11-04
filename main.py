import os
import json
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from io import BytesIO

app = FastAPI(title="AGRION TECHS - JD-CropAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/content/AGRION_RENDER/JD_CropAI_Model.pth"
DATABASE_PATH = "/content/AGRION_RENDER/JD_CropAI_Disease_Database.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

class_names = checkpoint["class_names"]

with open(DATABASE_PATH, "r") as f:
    disease_database = json.load(f)

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():
    return {
        "project": "AGRION TECHS",
        "developer": "JD VEXOR",
        "status": "Running",
        "model_classes": len(class_names)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database_categories": len(disease_database),
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg", "image/png", "image/webp"
    ]:
        raise HTTPException(400, "Upload JPG, PNG or WEBP.")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image must be below 10 MB.")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image file.")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    category = class_names[predicted.item()]
    confidence_value = round(confidence.item() * 100, 2)

    info = disease_database.get(category, {})

    crop = category.split("___")[0].replace("_", " ")
    disease = category.split("___")[-1].replace("_", " ")

    return {
        "project": "AGRION TECHS",
        "crop": crop,
        "prediction": disease,
        "confidence": confidence_value,
        "symptoms": info.get("symptoms", "Not available"),
        "causes": info.get("causes", "Not available"),
        "prevention": info.get("prevention", "Not available"),
        "care": info.get("care", "Not available"),
        "warning": "AI prediction only. Confirm with an agricultural expert."
    }
