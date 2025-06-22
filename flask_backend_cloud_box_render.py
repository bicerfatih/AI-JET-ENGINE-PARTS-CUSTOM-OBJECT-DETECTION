from flask import Flask, request, jsonify, render_template, make_response
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import io
from PIL import Image

# NEW IMPORTS for local .exe execution
import subprocess
import os

app = Flask(__name__)
CORS(app)

# Load the YOLO model
# IMPORTANT: Ensure EEMC_ForsaTek_XLarge.pt is in the same directory as this Python file
model = YOLO("EEMC_ForsaTek_Medium.pt")

# --- Configuration for Local Windows .exe Launch (Ignored by Render) ---
# Create a 'tools' folder in your project root and place your .exe file there.
# Example: YourProjectRoot/tools/MyExternalApp.exe
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')
# !!! IMPORTANT: Replace 'MyExternalApp.exe' with the exact name of your .exe file !!!
MY_EXE_PATH = os.path.join(TOOLS_DIR, 'MyExternalApp.exe')

# Optional: This block creates a dummy .bat file to act as an .exe for local testing
# if your real .exe isn't there. REMOVE or comment out for production or if you
# always have your real .exe in place.
if os.name == 'nt': # Check if the operating system is Windows
    if not os.path.exists(MY_EXE_PATH):
        try:
            os.makedirs(TOOLS_DIR, exist_ok=True)
            with open(MY_EXE_PATH, 'w') as f:
                f.write('@echo off\n')
                f.write('echo This is a dummy Windows application running from Python!\n')
                f.write('timeout /t 3 > nul\n') # Windows command to wait for 3 seconds
                f.write('echo Dummy application finished.\n')
            print(f"Created a dummy executable at: {MY_EXE_PATH} for local Windows testing.")
        except Exception as e:
            print(f"Could not create dummy .exe for local testing: {e}")
            print("Please ensure you have a real .exe file named exactly as set in MY_EXE_PATH for testing.")


@app.route('/')
def index():
    # Make sure 'index_cloud_box_render_camoffset_screenrotate_pdfmanual_pdfjs.html' exists in /templates/
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

    results = model(frame, conf=0.6)[0]
    labels = []

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        labels.append(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        if idx == 0:
            break

    success, img_encoded = cv2.imencode('.jpg', frame)
    if not success:
        return jsonify({'error': 'Image encoding failed'}), 500

    response = make_response(img_encoded.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('X-Labels', "|".join(labels))
    response.headers.set('Cache-Control', 'no-store')
    return response

# --- NEW ROUTE for Local Windows .exe Execution ---
@app.route('/launch_windows_app', methods=['POST'])
def launch_windows_app():
    """
    Launches a specified Windows .exe application on the server (your local Windows machine).
    This route is only intended to be used when running the Flask app locally on Windows.
    It will not work (and should be ignored) on Render.
    """
    # Ensure this function only attempts to run the .exe on Windows
    if os.name != 'nt':
        return jsonify({"status": "error", "message": "This feature is only available on Windows."}), 400

    if not os.path.exists(MY_EXE_PATH):
        print(f"Attempted to launch {MY_EXE_PATH} but it does not exist.")
        return jsonify({"status": "error", "message": f"Application not found at {os.path.basename(MY_EXE_PATH)}."}), 404

    try:
        print(f"Launching {os.path.basename(MY_EXE_PATH)} locally...")
        # subprocess.Popen() launches the application without blocking the Flask server.
        # This is ideal for GUI applications.
        subprocess.Popen([MY_EXE_PATH])
        print(f"{os.path.basename(MY_EXE_PATH)} launched successfully.")
        return jsonify({"status": "success", "message": f"{os.path.basename(MY_EXE_PATH)} launched on your local machine."})
    except Exception as e:
        print(f"Failed to launch {os.path.basename(MY_EXE_PATH)}: {e}")
        return jsonify({"status": "error", "message": f"Failed to launch application: {str(e)}"}), 500

# --- Main execution block ---
if __name__ == "__main__":
    # This block runs ONLY when you execute this Python script directly (e.g., 'python your_app.py' in PyCharm).
    # It does NOT run when Gunicorn is managing the app on Render or when PyInstaller creates the .exe.
    print("Running Flask app locally with Werkzeug development server.")
    # For local development and .exe packaging:
    # 1. Bind to localhost (127.0.0.1) for local access.
    # 2. Use HTTP (remove ssl_context) as Render handles HTTPS, and ad-hoc certs complicate .exe.
    # 3. Set debug=False for more production-like behavior and for .exe.
    app.run(host="127.0.0.1", port=5001, debug=False)
