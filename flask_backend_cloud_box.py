from flask import Flask, request, jsonify, render_template, make_response
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the YOLO model
model = YOLO("best_EEMC(5).pt")

@app.route('/')
def index():
    return render_template('index_cloud_box_render.html')  # Make sure this exists in /templates/

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.6)[0]
    labels = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        labels.append(label)

        # Draw bounding boxes and label text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Encode the result frame as JPEG
    success, img_encoded = cv2.imencode('.jpg', frame)
    if not success:
        return jsonify({'error': 'Image encoding failed'}), 500

    # Create a proper HTTP response with headers
    response = make_response(img_encoded.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('X-Labels', "|".join(labels))
    response.headers.set('Cache-Control', 'no-store')  # Prevent browser caching
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
