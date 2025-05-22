from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
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
    return render_template('index_cloud_box.html')  # Make sure this file exists in your templates folder

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

    # Draw bounding boxes and collect labels
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        labels.append(label)

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Send image and labels in response
    response = send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg'
    )
    response.headers["X-Labels"] = "|".join(labels)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
app = Flask(__name__)
CORS(app)

# Load the YOLO model
model = YOLO("best_EEMC(5).pt")

@app.route('/')
def index():
    return render_template('index_cloud_box.html')  # Make sure this file exists in your templates folder

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

    # Draw bounding boxes and collect labels
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        labels.append(label)

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Send image and labels in response
    response = send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg'
    )
    response.headers["X-Labels"] = "|".join(labels)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
