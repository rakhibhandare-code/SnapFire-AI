import sys
import cv2
from collections import deque
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from ultralytics import YOLO


# ============================================================
# SNAPFIRE AI MODULES
# ============================================================

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
# DASHBOARD
# ============================================================

class SnapFireDashboard(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "SnapFire AI - Multimodal Fire Detection"
        )

        self.resize(1500, 900)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print("Loading YOLO V2...")

        self.model = YOLO(
            MODEL_PATH
        )

        print("YOLO V2 loaded successfully.")

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():

            print("WARNING: Camera could not be opened.")

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
        # SYSTEM DATA
        # ----------------------------------------------------

        self.last_status = None

        self.final_status = "NORMAL"

        self.vision_status = "NORMAL"

        self.sensor_status = "NORMAL"

        self.fire_confidence = 0.0

        self.smoke_confidence = 0.0

        self.sensor_score = 0.0

        self.fusion_score = 0.0

        self.frame_count = 0

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

        main_layout.setSpacing(10)


        # ====================================================
        # HEADER
        # ====================================================

        title = QLabel(
            "SNAPFIRE AI"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setFont(
            QFont(
                "Arial",
                30,
                QFont.Bold
            )
        )

        subtitle = QLabel(
            "On-Device Multimodal Fire Detection "
            "& False-Alarm Reduction"
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
            title
        )

        main_layout.addWidget(
            subtitle
        )


        # ====================================================
        # TOP GRID
        # ====================================================

        top_grid = QGridLayout()

        top_grid.setSpacing(10)


        # ====================================================
        # CAMERA CARD
        # ====================================================

        camera_card = self.create_card()

        camera_layout = QVBoxLayout(
            camera_card
        )

        camera_title = QLabel(
            "LIVE CAMERA • YOLO V2"
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
            850,
            500
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setStyleSheet(
            "background-color: #101010;"
        )

        camera_layout.addWidget(
            camera_title
        )

        camera_layout.addWidget(
            self.camera_label
        )


        top_grid.addWidget(
            camera_card,
            0,
            0,
            2,
            2
        )


        # ====================================================
        # SYSTEM STATUS CARD
        # ====================================================

        status_card = self.create_card()

        status_layout = QVBoxLayout(
            status_card
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
                30,
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

        vision_title = QLabel(
            "VISION EVIDENCE"
        )

        vision_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        self.vision_status_label = QLabel(
            "Status: NORMAL"
        )

        self.fire_conf_label = QLabel(
            "Fire confidence: 0.00"
        )

        self.smoke_conf_label = QLabel(
            "Smoke confidence: 0.00"
        )

        self.temporal_label = QLabel(
            "Temporal confirmation: 0/10"
        )

        status_layout.addWidget(
            vision_title
        )

        status_layout.addWidget(
            self.vision_status_label
        )

        status_layout.addWidget(
            self.fire_conf_label
        )

        status_layout.addWidget(
            self.smoke_conf_label
        )

        status_layout.addWidget(
            self.temporal_label
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

        self.sensor_status_label = QLabel(
            "Status: NORMAL"
        )

        self.sensor_score_label = QLabel(
            "Evidence score: 0.00"
        )

        self.sensor_source_label = QLabel(
            "Source: DEMO / SIMULATED"
        )

        status_layout.addWidget(
            sensor_title
        )

        status_layout.addWidget(
            self.sensor_status_label
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

        self.fusion_reason_label = QLabel(
            "Reason: No significant evidence"
        )

        self.fusion_score_label = QLabel(
            "Fusion score: 0.00"
        )

        status_layout.addWidget(
            fusion_title
        )

        status_layout.addWidget(
            self.fusion_reason_label
        )

        status_layout.addWidget(
            self.fusion_score_label
        )


        # ----------------------------------------------------
        # DEMO SCENARIO
        # ----------------------------------------------------

        scenario_title = QLabel(
            "DEMO SENSOR SCENARIO"
        )

        scenario_title.setFont(
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
            scenario_title
        )

        status_layout.addWidget(
            self.scenario_combo
        )


        # ----------------------------------------------------
        # RESET
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


        top_grid.addWidget(
            status_card,
            0,
            2
        )


        # ====================================================
        # EVENT HISTORY CARD
        # ====================================================

        history_card = self.create_card()

        history_layout = QVBoxLayout(
            history_card
        )

        history_title = QLabel(
            "EVENT HISTORY"
        )

        history_title.setFont(
            QFont(
                "Arial",
                16,
                QFont.Bold
            )
        )

        self.event_list = QListWidget()

        self.event_list.setMinimumHeight(
            170
        )

        history_layout.addWidget(
            history_title
        )

        history_layout.addWidget(
            self.event_list
        )


        top_grid.addWidget(
            history_card,
            1,
            2
        )


        main_layout.addLayout(
            top_grid
        )


        # ====================================================
        # BOTTOM INFORMATION
        # ====================================================

        info_layout = QHBoxLayout()

        # ----------------------------------------------------
        # MODEL INFO
        # ----------------------------------------------------

        model_card = self.create_card()

        model_layout = QVBoxLayout(
            model_card
        )

        model_title = QLabel(
            "MODEL INFORMATION"
        )

        model_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        model_layout.addWidget(
            model_title
        )

        model_layout.addWidget(
            QLabel(
                "Vision: YOLO11n"
            )
        )

        model_layout.addWidget(
            QLabel(
                "Classes: Fire / Smoke"
            )
        )

        model_layout.addWidget(
            QLabel(
                "Temporal window: 10 frames"
            )
        )

        model_layout.addWidget(
            QLabel(
                "Confirmation: 6 positive frames"
            )
        )

        info_layout.addWidget(
            model_card
        )


        # ----------------------------------------------------
        # SYSTEM ARCHITECTURE
        # ----------------------------------------------------

        architecture_card = self.create_card()

        architecture_layout = QVBoxLayout(
            architecture_card
        )

        architecture_title = QLabel(
            "AI PIPELINE"
        )

        architecture_title.setFont(
            QFont(
                "Arial",
                14,
                QFont.Bold
            )
        )

        architecture_layout.addWidget(
            architecture_title
        )

        architecture_layout.addWidget(
            QLabel(
                "Camera → YOLO V2 → Temporal Filter"
            )
        )

        architecture_layout.addWidget(
            QLabel(
                "Sensors → Sensor Evidence"
            )
        )

        architecture_layout.addWidget(
            QLabel(
                "Vision + Sensors → Fusion Engine"
            )
        )

        architecture_layout.addWidget(
            QLabel(
                "Fusion Engine → Final System Status"
            )
        )

        info_layout.addWidget(
            architecture_card
        )


        main_layout.addLayout(
            info_layout
        )


        # ====================================================
        # FOOTER
        # ====================================================

        footer = QLabel(
            "SnapFire AI • Edge/On-Device Prototype • "
            "Not safety certified"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        footer.setStyleSheet(
            "color: #666;"
        )

        main_layout.addWidget(
            footer
        )


    # ========================================================
    # CREATE CARD
    # ========================================================

    def create_card(self):

        card = QFrame()

        card.setFrameShape(
            QFrame.StyledPanel
        )

        card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #d5d5d5;
                border-radius: 8px;
            }

            QLabel {
                border: none;
            }

            QListWidget {
                border: 1px solid #dddddd;
                background-color: #fafafa;
            }

            QComboBox {
                padding: 7px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }

            QPushButton {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f5f5f5;
            }

            QPushButton:hover {
                background-color: #e9e9e9;
            }
            """
        )

        return card


    # ========================================================
    # GET SENSOR VALUES
    # ========================================================

    def get_sensor_values(self):

        scenario = (
            self.scenario_combo.currentText()
        )

        return SCENARIOS[
            scenario
        ]


    # ========================================================
    # CAMERA PROCESSING
    # ========================================================

    def update_frame(self):

        self.frame_count += 1

        if not self.cap.isOpened():

            self.camera_label.setText(
                "Camera unavailable"
            )

            return


        ret, frame = self.cap.read()

        if not ret:
            return


        # ====================================================
        # YOLO INFERENCE
        # ====================================================

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


        # ====================================================
        # TEMPORAL FILTER
        # ====================================================

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


        # ====================================================
        # VISION STATUS
        # ====================================================

        if fire_confirmed:

            self.vision_status = "FIRE"

        elif smoke_confirmed:

            self.vision_status = "SMOKE"

        else:

            self.vision_status = "NORMAL"


        # ====================================================
        # SENSOR PROCESSING
        # ====================================================

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


        # ====================================================
        # FUSION
        # ====================================================

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


        new_status = (
            fusion_result["status"]
        )

        # ====================================================
        # SAVE CURRENT DATA
        # ====================================================

        self.fire_confidence = (
            fire_confidence
        )

        self.smoke_confidence = (
            smoke_confidence
        )

        self.fusion_score = (
            fusion_result["confidence"]
        )

        self.final_status = (
            new_status
        )


        # ====================================================
        # EVENT HISTORY
        # ====================================================

        if (
            self.last_status is None
            or new_status != self.last_status
        ):

            self.add_event(
                new_status,
                fusion_result["reason"]
            )

            self.last_status = (
                new_status
            )


        # ====================================================
        # UPDATE UI
        # ====================================================

        self.update_ui(
            fire_count,
            smoke_count,
            fusion_result
        )


        # ====================================================
        # DRAW CAMERA
        # ====================================================

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
    # UPDATE UI
    # ========================================================

    def update_ui(
        self,
        fire_count,
        smoke_count,
        fusion_result
    ):

        self.status_label.setText(
            self.final_status
        )


        self.vision_status_label.setText(
            f"Status: {self.vision_status}"
        )


        self.fire_conf_label.setText(
            f"Fire confidence: "
            f"{self.fire_confidence:.2f}"
        )


        self.smoke_conf_label.setText(
            f"Smoke confidence: "
            f"{self.smoke_confidence:.2f}"
        )


        self.temporal_label.setText(
            f"Temporal confirmation: "
            f"Fire {fire_count}/10 | "
            f"Smoke {smoke_count}/10"
        )


        self.sensor_status_label.setText(
            f"Status: {self.sensor_status}"
        )


        self.sensor_score_label.setText(
            f"Evidence score: "
            f"{self.sensor_score:.2f}"
        )


        self.fusion_reason_label.setText(
            "Reason: "
            + fusion_result["reason"]
        )


        self.fusion_score_label.setText(
            f"Fusion score: "
            f"{self.fusion_score:.2f}"
        )


        # ====================================================
        # STATUS STYLE
        # ====================================================

        if self.final_status == "FIRE":

            self.status_label.setStyleSheet(
                """
                color: red;
                font-weight: bold;
                """
            )

        elif self.final_status == "WARNING":

            self.status_label.setStyleSheet(
                """
                color: orange;
                font-weight: bold;
                """
            )

        elif self.final_status == "FALSE ALARM":

            self.status_label.setStyleSheet(
                """
                color: #cc8800;
                font-weight: bold;
                """
            )

        else:

            self.status_label.setStyleSheet(
                """
                color: green;
                font-weight: bold;
                """
            )


    # ========================================================
    # EVENT LOGGER
    # ========================================================

    def add_event(
        self,
        status,
        reason
    ):

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )


        text = (
            f"{timestamp}   "
            f"{status}   |   "
            f"{reason}"
        )


        self.event_list.insertItem(
            0,
            text
        )


        # Keep last 20 events

        while self.event_list.count() > 20:

            self.event_list.takeItem(
                self.event_list.count() - 1
            )


    # ========================================================
    # CHANGE SCENARIO
    # ========================================================

    def change_scenario(self):

        self.reset_history()

        scenario = (
            self.scenario_combo.currentText()
        )

        self.add_event(
            "DEMO",
            f"Scenario changed to {scenario}"
        )


    # ========================================================
    # RESET TEMPORAL HISTORY
    # ========================================================

    def reset_history(self):

        self.fire_history.clear()

        self.smoke_history.clear()


        self.add_event(
            "RESET",
            "Temporal detection history cleared"
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.timer.stop()

        if self.cap.isOpened():

            self.cap.release()

        event.accept()


# ============================================================
# START APPLICATION
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