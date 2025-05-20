from flask import Flask, Response, jsonify, render_template
import cv2
import torch
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
MODEL_PATH = "best_EEMC(5).pt"
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 0 if necessary


def generate_frames():
    """ Video streaming generator for Flask """
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection
        results = model(frame, conf=0.6)[0]

        detected_labels = []

        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            detected_labels.append(f"{label} ({confidence:.2f})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """ Serve the main webpage """
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    """ Route for video stream """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_labels')
def get_labels():
    """ API to return detected labels dynamically """
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Run YOLO detection
    results = model(frame, conf=0.6)[0]

    detected_labels = list(set([results.names[int(box.cls[0])] for box in results.boxes]))

    return jsonify({"detected_labels": detected_labels})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)