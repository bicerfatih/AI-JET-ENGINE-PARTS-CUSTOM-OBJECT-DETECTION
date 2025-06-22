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
# IMPORTANT: Ensure EEMC_ForsaTek2.pt is in the same directory as app.py
model = YOLO("EEMC_ForsaTek_Large.pt")

@app.route('/')
def index():
    # Make sure 'index_cloud_box_render_camoffset_screenrotate_pdfmanual.html' exists in /templates/
    return render_template('index_cloud_box_render_camoffset_screenrotate_pdfmanual_pdfjs.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    Handles image detection requests.
    Receives an image, performs YOLO detection, draws bounding boxes,
    and returns the processed image with detected labels in a header.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform detection with YOLO model
    # [0] to get the first (and only) result object
    results = model(frame, conf=0.6)[0]
    labels = []

    # Iterate through detected boxes (only process the first one as per original logic)
    for idx, box in enumerate(results.boxes):
        # Get bounding box coordinates and convert to integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Get label name and confidence score
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        labels.append(label)

        # Draw bounding box and label text on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 8) # Green rectangle
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3) # Green text

        if idx == 0:
            break  # Stop after the first detection as requested

    # Encode the modified frame as a JPEG image
    success, img_encoded = cv2.imencode('.jpg', frame)
    if not success:
        return jsonify({'error': 'Image encoding failed'}), 500

    # Create an HTTP response with the JPEG image bytes
    response = make_response(img_encoded.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    # Add the detected labels to a custom HTTP header for the frontend
    response.headers.set('X-Labels', "|".join(labels))
    # Prevent browser caching of detection results
    response.headers.set('Cache-Control', 'no-store')
    return response

if __name__ == "__main__":
    # Runs the Flask app on all available network interfaces (0.0.0.0)
    # on port 5001. 'ssl_context='adhoc'' generates a self-signed SSL certificate
    # for HTTPS, useful for development.
    app.run(host="0.0.0.0", port=5001, ssl_context='adhoc')
