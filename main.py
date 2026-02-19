"""Grounded SAM 2 — Open-vocabulary object detection + segmentation service."""

import base64
import io
import logging
import os
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("grounded-sam2")

PORT = int(os.environ.get("PORT", 8001))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", 0.3))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", 0.25))

app = FastAPI(title="Grounded SAM 2")

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------
_grounding_model = None
_sam2_predictor = None


def _load_grounding_dino():
    global _grounding_model
    if _grounding_model is not None:
        return _grounding_model
    log.info("Loading Grounding DINO …")
    from groundingdino.util.inference import load_model as _gd_load
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(
        repo_id="ShilongLiu/GroundingDINO",
        filename="GroundingDINO_SwinT_OGC.cfg.py",
    )
    weights_path = hf_hub_download(
        repo_id="ShilongLiu/GroundingDINO",
        filename="groundingdino_swint_ogc.pth",
    )
    _grounding_model = _gd_load(cfg_path, weights_path, device=DEVICE)
    log.info("Grounding DINO loaded.")
    return _grounding_model


def _load_sam2():
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor
    log.info("Loading SAM 2 …")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_large.pt",
    )
    cfg = "sam2.1_hiera_l.yaml"
    model = build_sam2(cfg, ckpt, device=DEVICE)
    _sam2_predictor = SAM2ImagePredictor(model)
    log.info("SAM 2 loaded.")
    return _sam2_predictor


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG
    prompts: list[str]  # text prompts, e.g. ["cat", "dog"]
    box_threshold: float = BOX_THRESHOLD
    text_threshold: float = TEXT_THRESHOLD


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] in pixels
    mask_rle: str  # base64-encoded binary mask (numpy bool array, uint8 PNG)


class PredictResponse(BaseModel):
    detections: list[Detection]
    width: int
    height: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "grounded-sam2",
        "device": DEVICE,
        "models_loaded": {
            "grounding_dino": _grounding_model is not None,
            "sam2": _sam2_predictor is not None,
        },
    }


def _encode_mask_png(mask: np.ndarray) -> str:
    """Encode a boolean mask as base64 PNG."""
    from PIL import Image
    img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    from PIL import Image
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Cannot decode image")

    image_np = np.array(pil_image)
    h, w = image_np.shape[:2]

    # --- Grounding DINO ---
    from groundingdino.util.inference import predict as gd_predict
    import groundingdino.datasets.transforms as T

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(pil_image, None)

    caption = ". ".join(req.prompts) + "."

    gd_model = _load_grounding_dino()
    boxes, logits, phrases = gd_predict(
        model=gd_model,
        image=image_transformed,
        caption=caption,
        box_threshold=req.box_threshold,
        text_threshold=req.text_threshold,
        device=DEVICE,
    )

    if len(boxes) == 0:
        return PredictResponse(detections=[], width=w, height=h)

    # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
    boxes_np = boxes.cpu().numpy()
    boxes_pixel = np.zeros_like(boxes_np)
    boxes_pixel[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
    boxes_pixel[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
    boxes_pixel[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
    boxes_pixel[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h

    # --- SAM 2 ---
    sam_predictor = _load_sam2()
    sam_predictor.set_image(image_np)

    input_boxes = torch.tensor(boxes_pixel, dtype=torch.float32, device=DEVICE)
    masks, scores, _ = sam_predictor.predict(
        box=input_boxes,
        multimask_output=False,
    )

    # masks shape: (N, 1, H, W) or (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    masks_np = masks.cpu().numpy().astype(bool)

    detections = []
    for i in range(len(boxes_pixel)):
        detections.append(Detection(
            label=phrases[i],
            confidence=float(logits[i]),
            box=boxes_pixel[i].tolist(),
            mask_rle=_encode_mask_png(masks_np[i]),
        ))

    return PredictResponse(detections=detections, width=w, height=h)


if __name__ == "__main__":
    log.info(f"Starting Grounded SAM 2 on port {PORT}, device={DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
