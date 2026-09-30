import sys
import os
import cv2
import joblib
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QComboBox
)

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"F:\SnapFire-AI"

SENSOR_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "sensor_model.pkl"
)

VISION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fire_smoke_yolo_small",
    "weights",
    "best.pt"
)

CAMERA_INDEX = 1


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading SnapFire AI models...")

sensor_model = joblib.load(SENSOR_MODEL_PATH)
vision_model = YOLO(VISION_MODEL_PATH)

print("Models loaded successfully.")


# ============================================================
# SENSOR SCENARIOS
# ============================================================

SCENARIOS = {
    "Normal Environment": {
        "Temperature": 25,
        "Smoke": 15,
        "Flame": 0,
        "Humidity": 50
    },

    "Possible Warning": {
        "Temperature": 33,
        "Smoke": 45,
        "Flame": 0,
        "Humidity": 35
    },

    "Fire Condition": {
        "Temperature": 40,
        "Smoke": 80,
        "Flame": 1,
        "Humidity": 22
    },

    "Smoke Without Flame": {
        "Temperature": 28,
        "Smoke": 90,
        "Flame": 0,
        "Humidity": 42
    }
}


# ============================================================
# SENSOR PREDICTION
# ============================================================

def sensor_prediction(values):

    df = pd.DataFrame([[
        values["Temperature"],
        values["Smoke"],
        values["Flame"],
        values["Humidity"]
    ]], columns=[
        "Temperature",
        "Smoke",
        "Flame",
        "Humidity"
    ])

    prediction = sensor_model.predict(df)[0]

    probabilities = sensor_model.predict_proba(df)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ============================================================
# FUSION ENGINE
# ============================================================

def fusion_decision(
    sensor_prediction,
    sensor_confidence,
    vision_prediction,
    vision_confidence
):

    sensor_confidence = sensor_confidence / 100.0

    # Strong visual fire
    if vision_prediction == "Fire" and vision_confidence >= 0.60:

        if sensor_prediction == "Fire":
            return "FIRE"

        elif sensor_prediction == "Warning":
            return "FIRE"

        elif sensor_prediction == "False Alarm":
            return "WARNING"

        else:
            return "WARNING"

    # Moderate visual fire
    if vision_prediction == "Fire" and vision_confidence >= 0.30:

        if sensor_prediction == "Fire":
            return "FIRE"

        elif sensor_prediction == "Warning":
            return "WARNING"

        elif sensor_prediction == "False Alarm":
            return "FALSE ALARM"

        else:
            return "WARNING"

    # Smoke detection
    if vision_prediction == "Smoke":

        if sensor_prediction == "Fire":
            return "FIRE"

        elif sensor_prediction == "Warning":
            return "WARNING"

        else:
            return "WARNING"

    # Sensor-only decision
    if sensor_prediction == "Fire":

        if sensor_confidence >= 0.80:
            return "FIRE"

        return "WARNING"

    if sensor_prediction == "Warning":
        return "WARNING"

    if sensor_prediction == "False Alarm":
        return "FALSE ALARM"

    return "NORMAL"


# ============================================================
# MAIN WINDOW
# ============================================================

class SnapFireDashboard(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "SnapFire AI - Multimodal Fire Detection"
        )

        self.setMinimumSize(1200, 750)

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        self.sensor_values = SCENARIOS[
            "Normal Environment"
        ]

        self.sensor_prediction = "Normal"
        self.sensor_confidence = 0

        self.vision_prediction = "None"
        self.vision_confidence = 0

        self.current_status = "NORMAL"

        self.build_ui()

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(30)


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        main_layout = QVBoxLayout(
            central
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = QLabel(
            "🔥 SNAPFIRE AI"
        )

        header.setFont(
            QFont("Segoe UI", 28, QFont.Bold)
        )

        header.setAlignment(
            Qt.AlignCenter
        )

        subtitle = QLabel(
            "On-Device Multimodal Fire Detection & False-Alarm Reduction"
        )

        subtitle.setAlignment(
            Qt.AlignCenter
        )

        subtitle.setFont(
            QFont("Segoe UI", 12)
        )

        main_layout.addWidget(header)

        main_layout.addWidget(subtitle)


        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

        content_layout = QHBoxLayout()

        main_layout.addLayout(
            content_layout
        )


        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_frame = QFrame()

        camera_frame.setFrameShape(
            QFrame.StyledPanel
        )

        camera_layout = QVBoxLayout(
            camera_frame
        )

        camera_title = QLabel(
            "LIVE CAMERA"
        )

        camera_title.setFont(
            QFont("Segoe UI", 16, QFont.Bold)
        )

        camera_layout.addWidget(
            camera_title
        )

        self.camera_label = QLabel(
            "Starting camera..."
        )

        self.camera_label.setMinimumSize(
            640,
            480
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setStyleSheet(
            """
            background-color: #111111;
            color: white;
            """
        )

        camera_layout.addWidget(
            self.camera_label
        )

        content_layout.addWidget(
            camera_frame,
            2
        )


        # ====================================================
        # RIGHT PANEL
        # ====================================================

        right_panel = QVBoxLayout()

        content_layout.addLayout(
            right_panel,
            1
        )


        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        status_frame = QFrame()

        status_layout = QVBoxLayout(
            status_frame
        )

        status_title = QLabel(
            "SYSTEM STATUS"
        )

        status_title.setFont(
            QFont("Segoe UI", 14, QFont.Bold)
        )

        self.status_label = QLabel(
            "NORMAL"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setFont(
            QFont("Segoe UI", 26, QFont.Bold)
        )

        self.status_label.setStyleSheet(
            """
            background-color: #1f7a1f;
            color: white;
            padding: 20px;
            border-radius: 10px;
            """
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            self.status_label
        )

        right_panel.addWidget(
            status_frame
        )


        # ----------------------------------------------------
        # SENSOR DATA
        # ----------------------------------------------------

        sensor_frame = QFrame()

        sensor_layout = QVBoxLayout(
            sensor_frame
        )

        sensor_title = QLabel(
            "SENSOR INTELLIGENCE"
        )

        sensor_title.setFont(
            QFont("Segoe UI", 14, QFont.Bold)
        )

        sensor_layout.addWidget(
            sensor_title
        )

        self.temperature_label = QLabel()

        self.smoke_label = QLabel()

        self.flame_label = QLabel()

        self.humidity_label = QLabel()

        for label in [
            self.temperature_label,
            self.smoke_label,
            self.flame_label,
            self.humidity_label
        ]:

            label.setFont(
                QFont("Segoe UI", 12)
            )

            sensor_layout.addWidget(
                label
            )

        right_panel.addWidget(
            sensor_frame
        )


        # ----------------------------------------------------
        # AI RESULTS
        # ----------------------------------------------------

        ai_frame = QFrame()

        ai_layout = QVBoxLayout(
            ai_frame
        )

        ai_title = QLabel(
            "AI FUSION"
        )

        ai_title.setFont(
            QFont("Segoe UI", 14, QFont.Bold)
        )

        ai_layout.addWidget(
            ai_title
        )

        self.sensor_result_label = QLabel()

        self.vision_result_label = QLabel()

        self.fusion_result_label = QLabel()

        ai_layout.addWidget(
            self.sensor_result_label
        )

        ai_layout.addWidget(
            self.vision_result_label
        )

        ai_layout.addWidget(
            self.fusion_result_label
        )

        right_panel.addWidget(
            ai_frame
        )


        # ----------------------------------------------------
        # SCENARIO SELECTOR
        # ----------------------------------------------------

        scenario_title = QLabel(
            "DEMO SCENARIO"
        )

        scenario_title.setFont(
            QFont("Segoe UI", 14, QFont.Bold)
        )

        right_panel.addWidget(
            scenario_title
        )

        self.scenario_combo = QComboBox()

        self.scenario_combo.addItems(
            SCENARIOS.keys()
        )

        self.scenario_combo.currentTextChanged.connect(
            self.change_scenario
        )

        right_panel.addWidget(
            self.scenario_combo
        )


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = QLabel(
            "SnapFire AI • Edge AI Prototype • "
            "Designed for Snapdragon-powered PCs"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        footer.setStyleSheet(
            """
            color: gray;
            padding: 10px;
            """
        )

        main_layout.addWidget(
            footer
        )

        self.update_sensor_display()


    # ========================================================
    # CHANGE SCENARIO
    # ========================================================

    def change_scenario(self, scenario):

        self.sensor_values = SCENARIOS[
            scenario
        ]

        self.update_sensor_display()


    # ========================================================
    # SENSOR DISPLAY
    # ========================================================

    def update_sensor_display(self):

        values = self.sensor_values

        self.temperature_label.setText(
            f"🌡 Temperature: {values['Temperature']} °C"
        )

        self.smoke_label.setText(
            f"💨 Smoke Level: {values['Smoke']}"
        )

        self.flame_label.setText(
            f"🔥 Flame: {values['Flame']}"
        )

        self.humidity_label.setText(
            f"💧 Humidity: {values['Humidity']} %"
        )


    # ========================================================
    # CAMERA + AI
    # ========================================================

    def update_frame(self):

        ret, frame = self.camera.read()

        if not ret:

            self.camera_label.setText(
                "Camera unavailable"
            )

            return


        # ----------------------------------------------------
        # SENSOR MODEL
        # ----------------------------------------------------

        (
            sensor_pred,
            sensor_conf
        ) = sensor_prediction(
            self.sensor_values
        )

        self.sensor_prediction = sensor_pred

        self.sensor_confidence = sensor_conf


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = vision_model(
            frame,
            conf=0.10,
            verbose=False
        )

        vision_pred = "None"

        vision_conf = 0

        for result in results:

            for box in result.boxes:

                cls_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    vision_model.names[
                        cls_id
                    ].lower()
                )

                if confidence > vision_conf:

                    vision_conf = confidence

                    if class_name == "fire":

                        vision_pred = "Fire"

                    elif class_name == "smoke":

                        vision_pred = "Smoke"


        self.vision_prediction = vision_pred

        self.vision_confidence = vision_conf


        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        final_status = fusion_decision(
            sensor_pred,
            sensor_conf,
            vision_pred,
            vision_conf
        )

        self.current_status = final_status


        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        annotated = results[0].plot()

        cv2.putText(
            annotated,
            f"Sensor: {sensor_pred} ({sensor_conf:.1f}%)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Vision: {vision_pred} ({vision_conf * 100:.1f}%)",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"FINAL: {final_status}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DISPLAY CAMERA
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        bytes_per_line = ch * w

        image = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            image
        )

        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )


        # ----------------------------------------------------
        # UPDATE AI DISPLAY
        # ----------------------------------------------------

        self.sensor_result_label.setText(
            f"Sensor AI: {sensor_pred} "
            f"({sensor_conf:.1f}%)"
        )

        self.vision_result_label.setText(
            f"Vision AI: {vision_pred} "
            f"({vision_conf * 100:.1f}%)"
        )

        self.fusion_result_label.setText(
            f"Fusion Decision: {final_status}"
        )

        self.update_status_color(
            final_status
        )


    # ========================================================
    # STATUS COLOR
    # ========================================================

    def update_status_color(
        self,
        status
    ):

        if status == "NORMAL":

            color = "#1f7a1f"

        elif status == "WARNING":

            color = "#b8860b"

        elif status == "FIRE":

            color = "#b22222"

        else:

            color = "#555555"


        self.status_label.setText(
            status
        )

        self.status_label.setStyleSheet(
            f"""
            background-color: {color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            """
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.timer.stop()

        self.camera.release()

        event.accept()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = SnapFireDashboard()

    window.show()

    sys.exit(
        app.exec()
    )