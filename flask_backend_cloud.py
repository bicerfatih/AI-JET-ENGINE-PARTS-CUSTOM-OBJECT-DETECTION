from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
MODEL_PATH = "best_EEMC(5).pt"
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    """ Serve the main webcam-based detection webpage """
    return render_template("index_cloud.html")

@app.route('/detect', methods=['POST'])
def detect():
    """ Endpoint to accept image from browser and run YOLO detection """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.5)[0]
    detected_labels = list(set([results.names[int(box.cls[0])] for box in results.boxes]))

    return jsonify({"detected": detected_labels})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)