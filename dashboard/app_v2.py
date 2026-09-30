import sys
import cv2

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO

# Import SnapFire modules
sys.path.append(r"F:\SnapFire-AI\src")

from sensor_evidence import evaluate_sensors
from fusion_engine_v2 import fuse_evidence


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"F:\SnapFire-AI\models"
    r"\fire_smoke_yolo_v2\weights\best.pt"
)

CAMERA_INDEX = 1

YOLO_CONFIDENCE = 0.25

WINDOW_SIZE = 10
MIN_POSITIVE_FRAMES = 6


# ============================================================
# DEMO SENSOR SCENARIOS
# ============================================================

SCENARIOS = {

    "Normal": {
        "temperature": 25,
        "humidity": 50,
        "tvoc": 100,
        "eco2": 450,
        "pm1": 2,
        "pm25": 3,
        "nc05": 10,
        "nc10": 2,
        "nc25": 1,
    },

    "Sensor Warning": {
        "temperature": 36,
        "humidity": 35,
        "tvoc": 1200,
        "eco2": 2200,
        "pm1": 120,
        "pm25": 150,
        "nc05": 150,
        "nc10": 100,
        "nc25": 50,
    },

    "Sensor Fire": {
        "temperature": 50,
        "humidity": 18,
        "tvoc": 3500,
        "eco2": 6000,
        "pm1": 1000,
        "pm25": 2000,
        "nc05": 5000,
        "nc10": 3000,
        "nc25": 1500,
    },

    "Multimodal Fire": {
        "temperature": 50,
        "humidity": 18,
        "tvoc": 3500,
        "eco2": 6000,
        "pm1": 1000,
        "pm25": 2000,
        "nc05": 5000,
        "nc10": 3000,
        "nc25": 1500,
    },
}


# ============================================================
# MAIN WINDOW
# ============================================================

class SnapFireDashboard(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "SnapFire AI - Multimodal Fire Detection"
        )

        self.resize(1400, 850)

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        self.model = YOLO(MODEL_PATH)

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        # ----------------------------------------------------
        # TEMPORAL HISTORY
        # ----------------------------------------------------

        self.fire_history = deque(
            maxlen=WINDOW_SIZE
        )

        self.smoke_history = deque(
            maxlen=WINDOW_SIZE
        )

        # ----------------------------------------------------
        # CURRENT DATA
        # ----------------------------------------------------

        self.fire_confidence = 0.0
        self.smoke_confidence = 0.0

        self.vision_status = "NORMAL"

        self.sensor_status = "NORMAL"
        self.sensor_score = 0.0

        self.final_status = "NORMAL"

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # TIMER
        # ----------------------------------------------------

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

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = QLabel(
            "SNAPFIRE AI"
        )

        header.setFont(
            QFont(
                "Arial",
                28,
                QFont.Bold
            )
        )

        header.setAlignment(
            Qt.AlignCenter
        )

        subtitle = QLabel(
            "On-Device Multimodal Fire Detection & "
            "False-Alarm Reduction"
        )

        subtitle.setAlignment(
            Qt.AlignCenter
        )

        subtitle.setFont(
            QFont(
                "Arial",
                12
            )
        )

        main_layout.addWidget(
            header
        )

        main_layout.addWidget(
            subtitle
        )


        # ====================================================
        # MAIN GRID
        # ====================================================

        grid = QGridLayout()

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

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
            QFont(
                "Arial",
                16,
                QFont.Bold
            )
        )

        self.camera_label = QLabel()

        self.camera_label.setMinimumSize(
            700,
            500
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setStyleSheet(
            "background-color: #111;"
        )

        camera_layout.addWidget(
            camera_title
        )

        camera_layout.addWidget(
            self.camera_label
        )

        grid.addWidget(
            camera_frame,
            0,
            0,
            3,
            2
        )


        # ====================================================
        # STATUS PANEL
        # ====================================================

        status_frame = QFrame()

        status_frame.setFrameShape(
            QFrame.StyledPanel
        )

        status_layout = QVBoxLayout(
            status_frame
        )

        status_title = QLabel(
            "SYSTEM STATUS"
        )

        status_title.setFont(
            QFont(
                "Arial",
                16,
                QFont.Bold
            )
        )

        self.status_label = QLabel(
            "NORMAL"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setFont(
            QFont(
                "Arial",
                28,
                QFont.Bold
            )
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            self.status_label
        )


        # ----------------------------------------------------
        # VISION
        # ----------------------------------------------------

        self.vision_label = QLabel(
            "Vision: NORMAL"
        )

        self.fire_label = QLabel(
            "Fire confidence: 0.00"
        )

        self.smoke_label = QLabel(
            "Smoke confidence: 0.00"
        )

        self.history_label = QLabel(
            "Temporal: 0/10"
        )

        status_layout.addWidget(
            self.vision_label
        )

        status_layout.addWidget(
            self.fire_label
        )

        status_layout.addWidget(
            self.smoke_label
        )

        status_layout.addWidget(
            self.history_label
        )


        # ----------------------------------------------------
        # SENSOR
        # ----------------------------------------------------

        sensor_title = QLabel(
            "SENSOR INTELLIGENCE"
        )

        sensor_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        self.sensor_label = QLabel(
            "Sensor: NORMAL"
        )

        self.sensor_score_label = QLabel(
            "Sensor evidence: 0.00"
        )

        self.sensor_source_label = QLabel(
            "Source: DEMO / SIMULATED"
        )

        status_layout.addWidget(
            sensor_title
        )

        status_layout.addWidget(
            self.sensor_label
        )

        status_layout.addWidget(
            self.sensor_score_label
        )

        status_layout.addWidget(
            self.sensor_source_label
        )


        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_title = QLabel(
            "AI FUSION"
        )

        fusion_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        self.fusion_reason = QLabel(
            "No significant evidence"
        )

        self.fusion_confidence = QLabel(
            "Fusion score: 0.00"
        )

        status_layout.addWidget(
            fusion_title
        )

        status_layout.addWidget(
            self.fusion_reason
        )

        status_layout.addWidget(
            self.fusion_confidence
        )


        # ====================================================
        # DEMO CONTROLS
        # ====================================================

        demo_title = QLabel(
            "DEMO SENSOR SCENARIO"
        )

        demo_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        self.scenario_combo = QComboBox()

        self.scenario_combo.addItems(
            list(
                SCENARIOS.keys()
            )
        )

        self.scenario_combo.currentTextChanged.connect(
            self.change_scenario
        )

        status_layout.addWidget(
            demo_title
        )

        status_layout.addWidget(
            self.scenario_combo
        )


        # ----------------------------------------------------
        # RESET BUTTON
        # ----------------------------------------------------

        reset_button = QPushButton(
            "Reset Temporal Detection"
        )

        reset_button.clicked.connect(
            self.reset_history
        )

        status_layout.addWidget(
            reset_button
        )

        status_layout.addStretch()


        grid.addWidget(
            status_frame,
            0,
            2,
            3,
            1
        )

        main_layout.addLayout(
            grid
        )


        # ====================================================
        # FOOTER
        # ====================================================

        footer = QLabel(
            "Prototype system — not safety certified"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        main_layout.addWidget(
            footer
        )


    # ========================================================
    # SENSOR SCENARIO
    # ========================================================

    def get_sensor_values(self):

        scenario = (
            self.scenario_combo.currentText()
        )

        return SCENARIOS[
            scenario
        ]


    # ========================================================
    # CAMERA UPDATE
    # ========================================================

    def update_frame(self):

        if not self.cap.isOpened():

            self.camera_label.setText(
                "Camera unavailable"
            )

            return


        ret, frame = self.cap.read()

        if not ret:

            return


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )


        fire_detected = False
        smoke_detected = False

        fire_confidence = 0.0
        smoke_confidence = 0.0


        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                if class_id == 0:

                    fire_detected = True

                    fire_confidence = max(
                        fire_confidence,
                        confidence
                    )

                elif class_id == 1:

                    smoke_detected = True

                    smoke_confidence = max(
                        smoke_confidence,
                        confidence
                    )


        # ----------------------------------------------------
        # TEMPORAL HISTORY
        # ----------------------------------------------------

        self.fire_history.append(
            fire_detected
        )

        self.smoke_history.append(
            smoke_detected
        )


        fire_count = sum(
            self.fire_history
        )

        smoke_count = sum(
            self.smoke_history
        )


        fire_confirmed = (
            len(self.fire_history)
            >= WINDOW_SIZE
            and fire_count
            >= MIN_POSITIVE_FRAMES
        )


        smoke_confirmed = (
            len(self.smoke_history)
            >= WINDOW_SIZE
            and smoke_count
            >= MIN_POSITIVE_FRAMES
        )


        # ----------------------------------------------------
        # VISION STATUS
        # ----------------------------------------------------

        if fire_confirmed:

            self.vision_status = "FIRE"

        elif smoke_confirmed:

            self.vision_status = "SMOKE"

        else:

            self.vision_status = "NORMAL"


        # ----------------------------------------------------
        # SENSOR
        # ----------------------------------------------------

        sensor_values = (
            self.get_sensor_values()
        )

        sensor_result = evaluate_sensors(
            **sensor_values
        )

        self.sensor_status = (
            sensor_result["status"]
        )

        self.sensor_score = (
            sensor_result["confidence"]
        )


        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_result = fuse_evidence(

            vision_status=self.vision_status,

            fire_confidence=fire_confidence,

            smoke_confidence=smoke_confidence,

            sensor_status=self.sensor_status,

            sensor_confidence=self.sensor_score,

            vision_confirmed=(
                fire_confirmed
                or smoke_confirmed
            )
        )


        self.final_status = (
            fusion_result["status"]
        )


        # ----------------------------------------------------
        # UPDATE UI
        # ----------------------------------------------------

        self.update_status_ui(
            fire_confidence,
            smoke_confidence,
            fire_count,
            smoke_count,
            fusion_result
        )


        # ----------------------------------------------------
        # CAMERA IMAGE
        # ----------------------------------------------------

        annotated = results[0].plot()

        rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = (
            rgb.shape
        )

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            rgb.data,
            width,
            height,
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


    # ========================================================
    # UPDATE STATUS UI
    # ========================================================

    def update_status_ui(
        self,
        fire_confidence,
        smoke_confidence,
        fire_count,
        smoke_count,
        fusion_result
    ):

        self.status_label.setText(
            self.final_status
        )

        self.vision_label.setText(
            f"Vision: {self.vision_status}"
        )

        self.fire_label.setText(
            f"Fire confidence: "
            f"{fire_confidence:.2f}"
        )

        self.smoke_label.setText(
            f"Smoke confidence: "
            f"{smoke_confidence:.2f}"
        )

        self.history_label.setText(
            f"Temporal: "
            f"Fire {fire_count}/10 | "
            f"Smoke {smoke_count}/10"
        )

        self.sensor_label.setText(
            f"Sensor: {self.sensor_status}"
        )

        self.sensor_score_label.setText(
            f"Sensor evidence: "
            f"{self.sensor_score:.2f}"
        )

        self.fusion_reason.setText(
            "Reason: "
            + fusion_result["reason"]
        )

        self.fusion_confidence.setText(
            f"Fusion score: "
            f"{fusion_result['confidence']:.2f}"
        )


        # ----------------------------------------------------
        # STATUS APPEARANCE
        # ----------------------------------------------------

        if self.final_status == "FIRE":

            self.status_label.setStyleSheet(
                "color: red; font-weight: bold;"
            )

        elif self.final_status == "WARNING":

            self.status_label.setStyleSheet(
                "color: orange; font-weight: bold;"
            )

        elif self.final_status == "FALSE ALARM":

            self.status_label.setStyleSheet(
                "color: #cc8800; font-weight: bold;"
            )

        else:

            self.status_label.setStyleSheet(
                "color: green; font-weight: bold;"
            )


    # ========================================================
    # CHANGE SCENARIO
    # ========================================================

    def change_scenario(self):

        self.reset_history()


    # ========================================================
    # RESET
    # ========================================================

    def reset_history(self):

        self.fire_history.clear()

        self.smoke_history.clear()


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.timer.stop()

        if self.cap.isOpened():

            self.cap.release()

        event.accept()


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = SnapFireDashboard()

    window.show()

    sys.exit(
        app.exec()
    )