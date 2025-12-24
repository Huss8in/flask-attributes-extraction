from flask import Flask, request, jsonify
import torch
from transformers import AutoModel, AutoConfig
from dotenv import load_dotenv
import os
import numpy as np

# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env")

# ------------------------------------------------------------------
# FORCE GPU
# ------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available. GPU is required.")

device = torch.device("cuda:0")
torch.cuda.set_device(device)

print(f"[INFO] Forced device: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------------
print("[INFO] Loading model config...")
config = AutoConfig.from_pretrained(
    "Blip-MAE-Botit/BlipMAEModel",
    trust_remote_code=True,
    token=HF_TOKEN,
)
config.model_type = "fashion"

# ------------------------------------------------------------------
# LOAD MODEL (GPU ONLY)
# ------------------------------------------------------------------
print("[INFO] Loading model weights...")
model = AutoModel.from_pretrained(
    "Blip-MAE-Botit/BlipMAEModel",
    trust_remote_code=True,
    config=config,
    token=HF_TOKEN
).to(device)

model.eval()

# hard check
assert next(model.parameters()).is_cuda, "Model is NOT on GPU"

print("[INFO] Model loaded on CUDA successfully.")

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def aggregate_predictions(results):
    if not results:
        return {}

    aggregated = {}
    attributes = results[0][0].keys()

    for attr in attributes:
        best_val, best_conf = None, -1
        for img_pred in results:
            val = img_pred[0][attr]["value"]
            conf = img_pred[0][attr]["confidence"]
            if conf > best_conf:
                best_conf = conf
                best_val = val

        aggregated[attr] = {
            "value": best_val,
            "confidence": float(best_conf)
        }

    return aggregated

# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "GradProject GPU Service",
        "gpu": torch.cuda.get_device_name(0),
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    images = data.get("images", [])
    description = data.get("descriptions", [""])[0]
    category = data.get("categories", [""])[0]
    attributes = data.get("attributes", [])

    if not images or not description or not category or not attributes:
        return jsonify({"error": "images, description, category, attributes required"}), 400

    print(f"[DEBUG] Images count: {len(images)}")

    with torch.inference_mode():
        results = model.generate(
            images_pth=images,
            descriptions=[description] * len(images),
            categories=[category] * len(images),
            attributes=attributes,
            return_confidences=True
        )

    results = convert_to_python(results)
    aggregated = aggregate_predictions(results)

    return jsonify({
        "description": description,
        "category": category,
        "attributes": aggregated
    })

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Flask running on GPU-only mode")
    app.run(host="0.0.0.0", port=5000, debug=False)
