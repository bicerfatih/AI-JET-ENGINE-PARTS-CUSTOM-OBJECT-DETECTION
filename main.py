import sys
import cv2
import torch
import pyperclip
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QTextEdit, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import AppKit
NSApplication = AppKit.NSApplication
NSObject = AppKit.NSObject
import objc

# Load YOLOv8 model (Replace with your downloaded model path)
MODEL_PATH = "best_EEMC2.pt"
model = YOLO(MODEL_PATH)

class AppDelegate(NSObject):
    def applicationSupportsSecureRestorableState_(self, app):
        return True  # Enable secure restorable state in macOS

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEMC - Engine Parts Real-Time Detection")
        self.setGeometry(100, 100, 1200, 600)
        self.detected_labels = set()  # Store last detected labels

        # Left Panel (Image Placeholder)
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(70, 800)
        self.image_label.setMaximumSize(300, 900)

        # Load the Image
        self.pixmap = QPixmap("EEMC_Banner.png")

        # Default Start Size
        default_width, default_height = 300, 870

        # Scale image dynamically with the panel size
        self.image_label.setPixmap(self.pixmap.scaled(
            default_width, default_height,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        ))

        self.image_label.setAlignment(Qt.AlignCenter)

        # Center Panel (Full-Screen Video Feed)
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Right Panel (Text Box for Labels + Copy Button)
        self.text_box = QTextEdit(self)
        self.text_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_box.setReadOnly(True)

        self.copy_button = QPushButton("Copy Labels")
        self.copy_button.clicked.connect(self.copy_labels)
        self.copy_button.setMinimumSize(50, 60)
        self.copy_button.setStyleSheet("font-size: 18px; padding: 10px;")

        # Buttons (Under Video)
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setMinimumSize(180, 60)
        self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")

        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setMinimumSize(180, 60)
        self.stop_button.setStyleSheet("font-size: 18px; padding: 10px;")

        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Layout Setup
        layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.image_label)

        center_panel = QVBoxLayout()
        center_panel.addWidget(self.video_label, stretch=1)
        center_panel.addWidget(self.start_button)
        center_panel.addWidget(self.stop_button)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.text_box)
        right_panel.addWidget(self.copy_button)

        layout.addLayout(left_panel, stretch=1)
        layout.addLayout(center_panel, stretch=2)
        layout.addLayout(right_panel, stretch=1)

        self.setLayout(layout)

        # Camera & Timer Setup
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)  # Open webcam
        self.timer.start(30)  # Refresh every 30ms

    def stop_camera(self):
        """ Stops the camera and releases resources """
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize frame to fit UI dynamically while maintaining aspect ratio
        frame = cv2.resize(frame, (self.video_label.width(), self.video_label.height()))

        # Run YOLO detection
        results = model(frame, conf=0.6)[0]

        new_detected_labels = set()

        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            new_detected_labels.add(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Only update text box if new labels are detected
        if new_detected_labels:
            self.detected_labels = new_detected_labels  # Update stored labels

        self.text_box.setText("\n".join(self.detected_labels))  # Show the last detected labels

        # Convert frame for PyQt5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def copy_labels(self):
        """ Copies detected labels to clipboard """
        text = self.text_box.toPlainText()
        pyperclip.copy(text)

if __name__ == "__main__":
    # macOS Secure Restorable State
    app_mac = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app_mac.setDelegate_(delegate)

    # PyQt Application
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())