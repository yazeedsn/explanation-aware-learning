"""
Simple FastAPI service: upload a chest X-ray image, get back the
predicted class for one disease plus a Grad-CAM heatmap overlay.

Configure via environment variables, then run:
    CHECKPOINT=runs/runs_alpha_1/pulmonary_fibrosis/best.pt DISEASE="Pleural fibrosis" \
        uvicorn api:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/predict -F "file=@chest_xray.png"

Docs: http://localhost:8000/docs
"""

import base64
import io
import os
from functools import lru_cache

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..core import ModelConfig, build_model
from ..core.losses import ExplanationLoss

class Settings(BaseModel):
    checkpoint: str = os.environ.get("CHECKPOINT", "runs/runs_alpha_1/pulmonary_fibrosis/best.pt")
    disease: str = os.environ.get("DISEASE", "Pleural fibrosis")
    img_size: int = int(os.environ.get("IMG_SIZE", 224))
    device: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    quantile: float = float(os.environ.get("QUANTILE", 1))
    score_mode: str = os.environ.get("SCORE_MODE", "sqr")  # must match what the checkpoint was trained with
    use_probs: bool = os.environ.get("USE_PROBS", "false").lower() == "true"


settings = Settings()
app = FastAPI(title="Chest X-ray Explanation API", version="0.1.0")


class PredictionResponse(BaseModel):
    disease: str
    predicted_class: str  # "positive" | "negative"
    probability: float  # P(positive), in [0, 1]
    heatmap_png_base64: str  # base64-encoded PNG: input image with the Grad-CAM heatmap overlaid


@lru_cache(maxsize=1)
def get_model():
    """Loaded once per process and cached -- not reloaded per request."""
    model = build_model(ModelConfig(pretrained=False, num_classes=2))
    state_dict = torch.load(settings.checkpoint, map_location=settings.device)
    model.load_state_dict(state_dict)
    model.to(settings.device)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_explainer() -> ExplanationLoss:
    return ExplanationLoss(quantile=settings.quantile, score_mode=settings.score_mode, use_probs=settings.use_probs)


def preprocess_image(raw_bytes: bytes) -> tuple:
    """Loads any PIL-readable image (PNG/JPG/etc.), converts to grayscale,
    resizes to img_size. Returns:
      - a (1, 3, H, W) float tensor in [0, 1] -- same preprocessing as
        BinaryDiseaseDataset (grayscale -> 3-channel expand, div by 255)
      - the resized grayscale numpy array, for rendering the overlay

    Note: this does NOT handle raw DICOM -- if you need to serve DICOM
    uploads directly, swap this for vinbig_prep.dicom_io.load_dicom_as_array
    on the raw bytes instead of PIL.
    """
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    image = image.resize((settings.img_size, settings.img_size))
    array = np.array(image, dtype=np.uint8)

    tensor = torch.from_numpy(array.copy()).float().div(255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).contiguous()  # (1, 3, H, W)
    return tensor, array


def compute_heatmap(explainer: ExplanationLoss, logits: torch.Tensor, feature_map: torch.Tensor, img_size: int) -> np.ndarray:
    """Full-resolution, normalized Grad-CAM heatmap, upsampled to img_size.
    """
    scores = explainer.classification_score(logits)
    gradients = explainer.gradcam_gradients(scores, feature_map, create_graph=False)
    weights = explainer.gradcam_weights(gradients)
    heatmap = explainer.gradcam_heatmap(weights, feature_map)
    heatmap = explainer.minmax_normalize(heatmap).detach()

    heatmap_up = F.interpolate(heatmap.unsqueeze(1), size=(img_size, img_size), mode="bilinear", align_corners=False)
    return heatmap_up.squeeze(0).squeeze(0).cpu().numpy()  # (H, W) in [0, 1]


def render_overlay(gray_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> bytes:
    """Overlays the heatmap (jet colormap) on the grayscale image, returns PNG bytes."""
    base = np.stack([gray_image] * 3, axis=-1).astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3]  # (H, W, 3) in [0, 1]
    overlay = (1 - alpha) * base + alpha * colored
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got content-type={file.content_type}")

    raw_bytes = await file.read()
    tensor, gray_image = preprocess_image(raw_bytes)
    tensor = tensor.to(settings.device)

    model = get_model()
    explainer = get_explainer()

    logits, feature_map = model(tensor)
    probs = F.softmax(logits, dim=1)
    prob_positive = probs[0, 1].item()
    predicted_class = "positive" if prob_positive >= 0.5 else "negative"

    heatmap = compute_heatmap(explainer, logits, feature_map, settings.img_size)
    heatmap_png = render_overlay(gray_image, heatmap)
    heatmap_b64 = base64.b64encode(heatmap_png).decode("utf-8")

    return PredictionResponse(
        disease=settings.disease,
        predicted_class=predicted_class,
        probability=prob_positive,
        heatmap_png_base64=heatmap_b64,
    )


@app.get("/health")
def health():
    return {"status": "ok", "disease": settings.disease, "checkpoint": settings.checkpoint, "device": settings.device}
