import os
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from segment_anything import sam_model_registry, SamPredictor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

app = Flask(__name__)
UPLOAD_FOLDER = 'static/outputs'
INPUT_PATH = os.path.join(UPLOAD_FOLDER, 'input.jpg')
DETECTION_PATH = os.path.join(UPLOAD_FOLDER, 'detection.jpg')
SEGMENTED_PATH = os.path.join(UPLOAD_FOLDER, 'segmented.jpg')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_final.pth"  # Ensure this path is correct
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cpu"
MetadataCatalog.get("cancer_detection_val").thing_classes = ["benign", "malignant"]
metadata = MetadataCatalog.get("cancer_detection_val")
d2_predictor = DefaultPredictor(cfg)

# Setup SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_model/sam_vit_b_01ec64.pth")  # Ensure this path is correct
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
sam_predictor = SamPredictor(sam)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    file.save(INPUT_PATH)

    img = cv2.imread(INPUT_PATH)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    outputs = d2_predictor(img)
    instances = outputs["instances"].to("cpu")

    scores = instances.scores.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    if len(scores) == 0:
        return jsonify({"error": "No cancerous region detected."}), 200

    best_idx = scores.argmax()
    best_instance = instances[best_idx : best_idx + 1]

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(best_instance)
    cv2.imwrite(DETECTION_PATH, out.get_image()[:, :, ::-1])

    sam_predictor.set_image(img)
    best_box = boxes[best_idx].astype(np.int32)
    input_box = np.array([best_box])
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False
    )
    mask = masks[0]
    masked_image = np.zeros_like(img)
    masked_image[mask] = img[mask]

    Image.fromarray(masked_image).save(SEGMENTED_PATH)

    # Return JSON with relative URLs for frontend to load
    return jsonify({
        "input_image": "/static/outputs/input.jpg",
        "detection_image": "/static/outputs/detection.jpg",
        "segmented_image": "/static/outputs/segmented.jpg"
    })

@app.route("/clear", methods=["GET"])
def clear():
    for f in [INPUT_PATH, DETECTION_PATH, SEGMENTED_PATH]:
        if os.path.exists(f):
            os.remove(f)
    return jsonify({"cleared": True})

if __name__ == "__main__":
    app.run(debug=True)