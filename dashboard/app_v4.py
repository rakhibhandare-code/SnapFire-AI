import sys
import os
import cv2
import pygame

from collections import deque
from datetime import datetime

from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QSizePolicy,
)


# ============================================================
# SNAPFIRE AI V5
# ============================================================

MODEL_PATH = (
    r"F:\SnapFire-AI\models\fire_smoke_yolo_v2"
    r"\weights\best.pt"
)

CAMERA_INDEX = 1

YOLO_CONFIDENCE = 0.25

WINDOW_SIZE = 10
MIN_POSITIVE_FRAMES = 6

WARNING_SOUND = (
    r"F:\SnapFire-AI\assets\warning.wav"
)

FIRE_SOUND = (
    r"F:\SnapFire-AI\assets\fire_alarm.wav"
)


# ============================================================
# COLORS
# ============================================================

BG = "#081018"
PANEL = "#101A24"
PANEL_LIGHT = "#16222E"
BORDER = "#263746"

TEXT = "#EAF2F8"
MUTED = "#8EA2B5"

GREEN = "#35D39A"
AMBER = "#F4B942"
RED = "#FF5361"
BLUE = "#55A8FF"


# ============================================================
# DEMO SENSOR SCENARIOS
# ============================================================

SCENARIOS = {

    "Normal": {
        "Temperature": 25.0,
        "Humidity": 50.0,
        "TVOC": 100.0,
        "eCO2": 450.0,
        "PM1.0": 2.0,
        "PM2.5": 3.0,
        "NC0.5": 10.0,
        "NC1.0": 2.0,
        "NC2.5": 1.0,
    },

    "Sensor Warning": {
        "Temperature": 36.0,
        "Humidity": 35.0,
        "TVOC": 1200.0,
        "eCO2": 2200.0,
        "PM1.0": 120.0,
        "PM2.5": 150.0,
        "NC0.5": 150.0,
        "NC1.0": 100.0,
        "NC2.5": 50.0,
    },

    "Sensor Fire": {
        "Temperature": 50.0,
        "Humidity": 18.0,
        "TVOC": 3500.0,
        "eCO2": 6000.0,
        "PM1.0": 1000.0,
        "PM2.5": 2000.0,
        "NC0.5": 5000.0,
        "NC1.0": 3000.0,
        "NC2.5": 1500.0,
    },

    "Multimodal Fire": {
        "Temperature": 50.0,
        "Humidity": 18.0,
        "TVOC": 3500.0,
        "eCO2": 6000.0,
        "PM1.0": 1000.0,
        "PM2.5": 2000.0,
        "NC0.5": 5000.0,
        "NC1.0": 3000.0,
        "NC2.5": 1500.0,
    },
}


# ============================================================
# HELPER
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


# ============================================================
# SENSOR EVIDENCE
# ============================================================

def evaluate_sensors(
    Temperature,
    Humidity,
    TVOC,
    eCO2,
    PM1_0,
    PM2_5,
    NC0_5,
    NC1_0,
    NC2_5,
):

    evidence = []
    score = 0.0

    # Temperature
    if Temperature >= 45:
        evidence.append("High temperature")
        score += 0.20

    elif Temperature >= 35:
        evidence.append("Elevated temperature")
        score += 0.10

    # Humidity
    if Humidity <= 20:
        evidence.append("Low humidity")
        score += 0.10

    elif Humidity <= 30:
        score += 0.05

    # TVOC
    if TVOC >= 3000:
        evidence.append("Very high TVOC")
        score += 0.20

    elif TVOC >= 1000:
        evidence.append("Elevated TVOC")
        score += 0.10

    # eCO2
    if eCO2 >= 5000:
        evidence.append("Very high eCO2")
        score += 0.20

    elif eCO2 >= 2000:
        evidence.append("Elevated eCO2")
        score += 0.10

    # PM2.5
    if PM2_5 >= 1000:
        evidence.append("Very high PM2.5")
        score += 0.15

    elif PM2_5 >= 100:
        evidence.append("Elevated PM2.5")
        score += 0.08

    # PM1
    if PM1_0 >= 500:
        evidence.append("Very high PM1.0")
        score += 0.10

    elif PM1_0 >= 100:
        evidence.append("Elevated PM1.0")
        score += 0.05

    # Particle counts
    if NC0_5 >= 1000:
        evidence.append("High NC0.5")
        score += 0.10

    elif NC0_5 >= 100:
        score += 0.04

    if NC1_0 >= 1000:
        evidence.append("High NC1.0")
        score += 0.08

    elif NC1_0 >= 100:
        score += 0.03

    if NC2_5 >= 500:
        evidence.append("High NC2.5")
        score += 0.07

    elif NC2_5 >= 50:
        score += 0.03

    score = clamp(score)

    # Strong sensor fire evidence
    if (
        Temperature >= 45
        and eCO2 >= 5000
        and PM2_5 >= 1000
    ):
        status = "FIRE"

    elif score >= 0.25:
        status = "WARNING"

    else:
        status = "NORMAL"

    if evidence:
        evidence_text = ", ".join(evidence)
    else:
        evidence_text = "No significant sensor anomaly"

    return status, score, evidence_text


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def fuse_evidence(
    vision_status,
    vision_confidence,
    sensor_status,
    sensor_confidence,
    temporal_confirmed,
):

    # --------------------------------------------------------
    # Vision + sensor agreement
    # --------------------------------------------------------

    if (
        vision_status == "FIRE"
        and temporal_confirmed
        and sensor_status == "FIRE"
    ):

        confidence = (
            0.45 * vision_confidence
            + 0.35 * sensor_confidence
            + 0.20
        )

        return (
            "FIRE",
            clamp(confidence),
            "Vision and sensor evidence agree",
        )

    # --------------------------------------------------------
    # Persistent vision + warning sensor
    # --------------------------------------------------------

    if (
        vision_status == "FIRE"
        and temporal_confirmed
        and sensor_status == "WARNING"
    ):

        confidence = (
            0.55 * vision_confidence
            + 0.25 * sensor_confidence
        )

        return (
            "WARNING",
            clamp(confidence),
            "Persistent visual fire evidence "
            "with sensor anomaly",
        )

    # --------------------------------------------------------
    # Persistent vision + normal sensor
    # --------------------------------------------------------

    if (
        vision_status == "FIRE"
        and temporal_confirmed
        and sensor_status == "NORMAL"
    ):

        confidence = (
            0.65 * vision_confidence
        )

        return (
            "WARNING",
            clamp(confidence),
            "Persistent visual fire evidence "
            "but sensors are normal",
        )

    # --------------------------------------------------------
    # Visual indication without confirmation
    # --------------------------------------------------------

    if vision_status in ["FIRE", "SMOKE"]:

        if sensor_status == "FIRE":

            return (
                "WARNING",
                clamp(sensor_confidence),
                "Sensor fire indication "
                "without visual confirmation",
            )

        if sensor_status == "WARNING":

            confidence = (
                0.5 * vision_confidence
                + 0.5 * sensor_confidence
            )

            return (
                "WARNING",
                clamp(confidence),
                "Visual indication with "
                "sensor anomaly",
            )

        return (
            "WARNING",
            clamp(0.5 * vision_confidence),
            "Visual fire indication "
            "but sensors are normal",
        )

    # --------------------------------------------------------
    # Sensor-only fire
    # --------------------------------------------------------

    if sensor_status == "FIRE":

        return (
            "WARNING",
            clamp(sensor_confidence),
            "Strong sensor evidence "
            "without visual confirmation",
        )

    # --------------------------------------------------------
    # Sensor warning
    # --------------------------------------------------------

    if sensor_status == "WARNING":

        return (
            "WARNING",
            clamp(sensor_confidence),
            "Sensor anomaly detected",
        )

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    return (
        "NORMAL",
        1.0,
        "No significant fire or smoke evidence",
    )


# ============================================================
# ALARM SYSTEM
# ============================================================

class AlarmSystem:

    def __init__(self):

        self.enabled = False
        self.current_alarm = None

        self.warning_sound = None
        self.fire_sound = None

        try:

            pygame.mixer.init()

            if os.path.exists(WARNING_SOUND):

                self.warning_sound = (
                    pygame.mixer.Sound(WARNING_SOUND)
                )

                self.warning_sound.set_volume(0.5)

            if os.path.exists(FIRE_SOUND):

                self.fire_sound = (
                    pygame.mixer.Sound(FIRE_SOUND)
                )

                self.fire_sound.set_volume(0.7)

            self.enabled = True

            print("Alarm system: ONLINE")

        except Exception as error:

            print(
                "Alarm system unavailable:",
                error
            )

    def update(self, status):

        if not self.enabled:
            return

        if status == "FIRE":

            if self.current_alarm != "FIRE":

                pygame.mixer.stop()

                if self.fire_sound:

                    self.fire_sound.play(
                        loops=-1
                    )

                self.current_alarm = "FIRE"

        elif status == "WARNING":

            if self.current_alarm != "WARNING":

                pygame.mixer.stop()

                if self.warning_sound:

                    self.warning_sound.play(
                        loops=-1
                    )

                self.current_alarm = "WARNING"

        else:

            if self.current_alarm is not None:

                pygame.mixer.stop()

                self.current_alarm = None

    def stop(self):

        if self.enabled:

            pygame.mixer.stop()

        self.current_alarm = None


# ============================================================
# CARD WIDGET
# ============================================================

class Card(QFrame):

    def __init__(self):

        super().__init__()

        self.setObjectName("Card")


# ============================================================
# MAIN WINDOW
# ============================================================

class SnapFireDashboard(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "SnapFire AI — Multimodal Fire Detection"
        )

        # ----------------------------------------------------
        # WINDOW
        # ----------------------------------------------------

        self.resize(1500, 900)

        self.setMinimumSize(
            1100,
            700
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print("Loading YOLO model...")

        self.model = YOLO(MODEL_PATH)

        print("YOLO model loaded.")

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        self.camera_online = (
            self.cap.isOpened()
        )

        if self.camera_online:
            print("Camera: ONLINE")
        else:
            print("Camera: OFFLINE")

        # ----------------------------------------------------
        # TEMPORAL BUFFER
        # ----------------------------------------------------

        self.temporal_buffer = deque(
            maxlen=WINDOW_SIZE
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.current_status = "NORMAL"
        self.previous_status = "NORMAL"

        self.current_vision = "NORMAL"
        self.current_vision_confidence = 0.0

        self.current_sensor_status = "NORMAL"
        self.current_sensor_score = 0.0

        self.current_reason = (
            "No significant fire or smoke evidence"
        )

        self.events = []

        # ----------------------------------------------------
        # SENSOR VALUES
        # ----------------------------------------------------

        self.sensor_values = (
            SCENARIOS["Normal"].copy()
        )

        # ----------------------------------------------------
        # ALARM
        # ----------------------------------------------------

        self.alarm = AlarmSystem()

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # CAMERA TIMER
        # ----------------------------------------------------

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.process_frame
        )

        self.timer.start(30)

    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(self):

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {BG};
            }}

            QWidget {{
                color: {TEXT};
                font-family: Segoe UI;
                font-size: 13px;
            }}

            QLabel {{
                color: {TEXT};
            }}

            QFrame#Card {{
                background-color: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}

            QComboBox {{
                background-color: {PANEL_LIGHT};
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 9px;
                color: {TEXT};
            }}

            QPushButton {{
                background-color: {PANEL_LIGHT};
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 9px 14px;
                color: {TEXT};
            }}

            QPushButton:hover {{
                background-color: #20303E;
            }}
            """
        )

        central = QWidget()

        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )

        root.setContentsMargins(
            18,
            14,
            18,
            12
        )

        root.setSpacing(12)

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        # Logo/title
        title_box = QVBoxLayout()

        title = QLabel(
            "🔥  SNAPFIRE AI"
        )

        title.setStyleSheet(
            """
            font-size: 26px;
            font-weight: 800;
            """
        )

        subtitle = QLabel(
            "On-Device Multimodal Fire Detection "
            "• False-Alarm Reduction"
        )

        subtitle.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 12px;
            """
        )

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addLayout(title_box)

        header.addStretch()

        # Status indicators
        self.ai_indicator = QLabel(
            "● AI ENGINE ONLINE"
        )

        self.ai_indicator.setStyleSheet(
            f"""
            color: {GREEN};
            font-weight: 600;
            """
        )

        self.camera_indicator = QLabel(
            "● CAMERA ONLINE"
            if self.camera_online
            else "● CAMERA OFFLINE"
        )

        self.camera_indicator.setStyleSheet(
            f"""
            color: {
                GREEN if self.camera_online else RED
            };
            font-weight: 600;
            """
        )

        self.sensor_indicator = QLabel(
            "● SENSOR LINK: DEMO"
        )

        self.sensor_indicator.setStyleSheet(
            f"""
            color: {AMBER};
            font-weight: 600;
            """
        )

        header.addWidget(
            self.ai_indicator
        )

        header.addSpacing(18)

        header.addWidget(
            self.camera_indicator
        )

        header.addSpacing(18)

        header.addWidget(
            self.sensor_indicator
        )

        root.addLayout(header)

        # ====================================================
        # MAIN UPPER AREA
        # ====================================================

        upper = QHBoxLayout()

        upper.setSpacing(12)

        # ====================================================
        # LARGE CAMERA PANEL
        # ====================================================

        camera_card = Card()

        camera_layout = QVBoxLayout(
            camera_card
        )

        camera_layout.setContentsMargins(
            12,
            10,
            12,
            12
        )

        camera_layout.setSpacing(8)

        camera_header = QHBoxLayout()

        camera_title = QLabel(
            "LIVE VISION ANALYSIS"
        )

        camera_title.setStyleSheet(
            """
            font-size: 14px;
            font-weight: 700;
            """
        )

        camera_header.addWidget(
            camera_title
        )

        camera_header.addStretch()

        self.camera_mode = QLabel(
            "YOLO V2 • LIVE"
        )

        self.camera_mode.setStyleSheet(
            f"""
            color: {BLUE};
            font-size: 11px;
            font-weight: 600;
            """
        )

        camera_header.addWidget(
            self.camera_mode
        )

        camera_layout.addLayout(
            camera_header
        )

        self.camera_label = QLabel()

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumSize(
            500,
            320
        )

        self.camera_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.camera_label.setStyleSheet(
            """
            background-color: #03070B;
            border-radius: 8px;
            """
        )

        self.camera_label.setText(
            "Waiting for camera..."
        )

        camera_layout.addWidget(
            self.camera_label
        )

        upper.addWidget(
            camera_card,
            7
        )

        # ====================================================
        # RIGHT INTELLIGENCE PANEL
        # ====================================================

        intelligence = QVBoxLayout()

        intelligence.setSpacing(9)

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        status_card = Card()

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            12,
            10,
            12,
            10
        )

        status_header = QLabel(
            "SYSTEM STATUS"
        )

        status_header.setStyleSheet(
            """
            font-weight: 700;
            font-size: 13px;
            """
        )

        status_layout.addWidget(
            status_header
        )

        self.final_status_label = QLabel(
            "NORMAL"
        )

        self.final_status_label.setAlignment(
            Qt.AlignCenter
        )

        self.final_status_label.setMinimumHeight(
            65
        )

        status_layout.addWidget(
            self.final_status_label
        )

        self.final_confidence_label = QLabel(
            "Fusion Confidence  •  100%"
        )

        self.final_confidence_label.setAlignment(
            Qt.AlignCenter
        )

        self.final_confidence_label.setStyleSheet(
            f"color: {MUTED};"
        )

        status_layout.addWidget(
            self.final_confidence_label
        )

        intelligence.addWidget(
            status_card,
            2
        )

        # ----------------------------------------------------
        # VISION + SENSOR EVIDENCE
        # ----------------------------------------------------

        evidence_card = Card()

        evidence_layout = QVBoxLayout(
            evidence_card
        )

        evidence_layout.setContentsMargins(
            12,
            10,
            12,
            10
        )

        evidence_title = QLabel(
            "EVIDENCE"
        )

        evidence_title.setStyleSheet(
            "font-weight: 700;"
        )

        evidence_layout.addWidget(
            evidence_title
        )

        evidence_grid = QGridLayout()

        # Vision
        vision_box = QVBoxLayout()

        vision_caption = QLabel(
            "VISION"
        )

        vision_caption.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            font-weight: 700;
            """
        )

        self.vision_value = QLabel(
            "NORMAL"
        )

        self.vision_value.setStyleSheet(
            "font-size: 16px; font-weight: 700;"
        )

        self.vision_conf = QLabel(
            "Fire: 0%"
        )

        self.vision_conf.setStyleSheet(
            f"color: {MUTED};"
        )

        vision_box.addWidget(
            vision_caption
        )

        vision_box.addWidget(
            self.vision_value
        )

        vision_box.addWidget(
            self.vision_conf
        )

        evidence_grid.addLayout(
            vision_box,
            0,
            0
        )

        # Sensor
        sensor_box = QVBoxLayout()

        sensor_caption = QLabel(
            "SENSOR"
        )

        sensor_caption.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            font-weight: 700;
            """
        )

        self.sensor_value = QLabel(
            "NORMAL"
        )

        self.sensor_value.setStyleSheet(
            "font-size: 16px; font-weight: 700;"
        )

        self.sensor_score = QLabel(
            "Score: 0%"
        )

        self.sensor_score.setStyleSheet(
            f"color: {MUTED};"
        )

        sensor_box.addWidget(
            sensor_caption
        )

        sensor_box.addWidget(
            self.sensor_value
        )

        sensor_box.addWidget(
            self.sensor_score
        )

        evidence_grid.addLayout(
            sensor_box,
            0,
            1
        )

        evidence_layout.addLayout(
            evidence_grid
        )

        intelligence.addWidget(
            evidence_card,
            2
        )

        # ----------------------------------------------------
        # TEMPORAL CONFIRMATION
        # ----------------------------------------------------

        temporal_card = Card()

        temporal_layout = QVBoxLayout(
            temporal_card
        )

        temporal_layout.setContentsMargins(
            12,
            9,
            12,
            9
        )

        temporal_header = QHBoxLayout()

        temporal_title = QLabel(
            "TEMPORAL CONFIRMATION"
        )

        temporal_title.setStyleSheet(
            "font-weight: 700;"
        )

        temporal_header.addWidget(
            temporal_title
        )

        temporal_header.addStretch()

        self.temporal_count = QLabel(
            "0 / 10"
        )

        self.temporal_count.setStyleSheet(
            f"""
            color: {BLUE};
            font-weight: 700;
            """
        )

        temporal_header.addWidget(
            self.temporal_count
        )

        temporal_layout.addLayout(
            temporal_header
        )

        self.temporal_state = QLabel(
            "● Waiting for persistent evidence"
        )

        self.temporal_state.setStyleSheet(
            f"color: {MUTED};"
        )

        temporal_layout.addWidget(
            self.temporal_state
        )

        intelligence.addWidget(
            temporal_card,
            1
        )

        # ----------------------------------------------------
        # AI FUSION
        # ----------------------------------------------------

        fusion_card = Card()

        fusion_layout = QVBoxLayout(
            fusion_card
        )

        fusion_layout.setContentsMargins(
            12,
            9,
            12,
            9
        )

        fusion_title = QLabel(
            "AI FUSION"
        )

        fusion_title.setStyleSheet(
            "font-weight: 700;"
        )

        fusion_layout.addWidget(
            fusion_title
        )

        self.fusion_reason = QLabel(
            "No significant fire or smoke evidence"
        )

        self.fusion_reason.setWordWrap(
            True
        )

        self.fusion_reason.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 11px;
            """
        )

        fusion_layout.addWidget(
            self.fusion_reason
        )

        intelligence.addWidget(
            fusion_card,
            1
        )

        upper.addLayout(
            intelligence,
            3
        )

        root.addLayout(
            upper,
            6
        )

        # ====================================================
        # LOWER AREA
        # ====================================================

        lower = QHBoxLayout()

        lower.setSpacing(12)

        # ====================================================
        # SENSOR TELEMETRY
        # ====================================================

        telemetry_card = Card()

        telemetry_layout = QVBoxLayout(
            telemetry_card
        )

        telemetry_layout.setContentsMargins(
            12,
            9,
            12,
            10
        )

        telemetry_title = QLabel(
            "SENSOR TELEMETRY"
        )

        telemetry_title.setStyleSheet(
            "font-weight: 700;"
        )

        telemetry_layout.addWidget(
            telemetry_title
        )

        telemetry_grid = QGridLayout()

        telemetry_grid.setSpacing(7)

        self.telemetry_labels = {}

        sensor_names = [
            "Temperature",
            "Humidity",
            "TVOC",
            "eCO2",
            "PM1.0",
            "PM2.5",
            "NC0.5",
            "NC1.0",
            "NC2.5",
        ]

        for index, name in enumerate(
            sensor_names
        ):

            row = index // 3
            col = index % 3

            label = QLabel(
                f"{name}: --"
            )

            label.setMinimumHeight(
                30
            )

            label.setStyleSheet(
                f"""
                background-color: {PANEL_LIGHT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 5px 8px;
                """
            )

            self.telemetry_labels[
                name
            ] = label

            telemetry_grid.addWidget(
                label,
                row,
                col
            )

        telemetry_layout.addLayout(
            telemetry_grid
        )

        lower.addWidget(
            telemetry_card,
            7
        )

        # ====================================================
        # DEMO CONTROL
        # ====================================================

        control_card = Card()

        control_layout = QVBoxLayout(
            control_card
        )

        control_layout.setContentsMargins(
            12,
            9,
            12,
            10
        )

        control_title = QLabel(
            "DEMO CONTROL"
        )

        control_title.setStyleSheet(
            "font-weight: 700;"
        )

        control_layout.addWidget(
            control_title
        )

        self.scenario_combo = QComboBox()

        self.scenario_combo.addItems(
            SCENARIOS.keys()
        )

        self.scenario_combo.currentTextChanged.connect(
            self.change_scenario
        )

        control_layout.addWidget(
            self.scenario_combo
        )

        buttons = QHBoxLayout()

        self.test_alarm_button = QPushButton(
            "🔊  TEST ALARM"
        )

        self.test_alarm_button.clicked.connect(
            self.test_alarm
        )

        self.stop_alarm_button = QPushButton(
            "🔇  STOP"
        )

        self.stop_alarm_button.clicked.connect(
            self.stop_alarm
        )

        buttons.addWidget(
            self.test_alarm_button
        )

        buttons.addWidget(
            self.stop_alarm_button
        )

        control_layout.addLayout(
            buttons
        )

        pipeline = QLabel(
            "CAMERA → YOLO → TEMPORAL → FUSION\n"
            "SENSORS → EVIDENCE → FUSION → ALARM"
        )

        pipeline.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            """
        )

        control_layout.addWidget(
            pipeline
        )

        lower.addWidget(
            control_card,
            3
        )

        root.addLayout(
            lower,
            2
        )

        # ====================================================
        # EVENT TIMELINE
        # ====================================================

        event_card = Card()

        event_layout = QVBoxLayout(
            event_card
        )

        event_layout.setContentsMargins(
            12,
            8,
            12,
            8
        )

        event_header = QHBoxLayout()

        event_title = QLabel(
            "EVENT TIMELINE"
        )

        event_title.setStyleSheet(
            "font-weight: 700;"
        )

        event_header.addWidget(
            event_title
        )

        event_header.addStretch()

        self.event_count_label = QLabel(
            "0 events"
        )

        self.event_count_label.setStyleSheet(
            f"color: {MUTED};"
        )

        event_header.addWidget(
            self.event_count_label
        )

        event_layout.addLayout(
            event_header
        )

        self.event_label = QLabel(
            "System initialized — waiting for events."
        )

        self.event_label.setWordWrap(
            True
        )

        self.event_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 11px;
            """
        )

        event_layout.addWidget(
            self.event_label
        )

        root.addWidget(
            event_card,
            1
        )

        # ====================================================
        # FOOTER
        # ====================================================

        footer = QLabel(
            "SNAPFIRE AI  •  EDGE / ON-DEVICE PROTOTYPE  •  "
            "NOT SAFETY CERTIFIED"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        footer.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            """
        )

        root.addWidget(
            footer
        )

        # Initial values
        self.update_telemetry()
        self.update_status_ui()

    # ========================================================
    # SCENARIO
    # ========================================================

    def change_scenario(self, scenario):

        if scenario not in SCENARIOS:
            return

        self.sensor_values = (
            SCENARIOS[scenario].copy()
        )

        self.update_telemetry()

    # ========================================================
    # TELEMETRY
    # ========================================================

    def update_telemetry(self):

        units = {
            "Temperature": "°C",
            "Humidity": "%",
            "TVOC": "ppb",
            "eCO2": "ppm",
            "PM1.0": "µg/m³",
            "PM2.5": "µg/m³",
            "NC0.5": "",
            "NC1.0": "",
            "NC2.5": "",
        }

        for name, label in (
            self.telemetry_labels.items()
        ):

            value = self.sensor_values[name]

            label.setText(
                f"{name}: {value:.1f} {units[name]}"
            )

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self):

        if not self.camera_online:
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        try:

            results = self.model(
                frame,
                conf=YOLO_CONFIDENCE,
                verbose=False
            )

        except Exception as error:

            print(
                "YOLO error:",
                error
            )

            return

        fire_confidence = 0.0
        smoke_confidence = 0.0

        annotated_frame = frame.copy()

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        for result in results:

            annotated_frame = result.plot()

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = str(
                    self.model.names[class_id]
                ).lower()

                if class_name == "fire":

                    fire_confidence = max(
                        fire_confidence,
                        confidence
                    )

                elif class_name == "smoke":

                    smoke_confidence = max(
                        smoke_confidence,
                        confidence
                    )

        # ----------------------------------------------------
        # VISION CLASSIFICATION
        # ----------------------------------------------------

        if fire_confidence >= YOLO_CONFIDENCE:

            vision_status = "FIRE"
            vision_confidence = fire_confidence
            positive = True

        elif smoke_confidence >= YOLO_CONFIDENCE:

            vision_status = "SMOKE"
            vision_confidence = smoke_confidence
            positive = True

        else:

            vision_status = "NORMAL"
            vision_confidence = 0.0
            positive = False

        # ----------------------------------------------------
        # TEMPORAL
        # ----------------------------------------------------

        self.temporal_buffer.append(
            positive
        )

        positive_frames = sum(
            self.temporal_buffer
        )

        temporal_confirmed = (
            positive_frames
            >= MIN_POSITIVE_FRAMES
        )

        # ----------------------------------------------------
        # SENSOR
        # ----------------------------------------------------

        (
            sensor_status,
            sensor_score,
            sensor_evidence,
        ) = evaluate_sensors(

            Temperature=self.sensor_values[
                "Temperature"
            ],

            Humidity=self.sensor_values[
                "Humidity"
            ],

            TVOC=self.sensor_values[
                "TVOC"
            ],

            eCO2=self.sensor_values[
                "eCO2"
            ],

            PM1_0=self.sensor_values[
                "PM1.0"
            ],

            PM2_5=self.sensor_values[
                "PM2.5"
            ],

            NC0_5=self.sensor_values[
                "NC0.5"
            ],

            NC1_0=self.sensor_values[
                "NC1.0"
            ],

            NC2_5=self.sensor_values[
                "NC2.5"
            ],
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        (
            final_status,
            final_confidence,
            reason,
        ) = fuse_evidence(

            vision_status=vision_status,

            vision_confidence=(
                vision_confidence
            ),

            sensor_status=sensor_status,

            sensor_confidence=sensor_score,

            temporal_confirmed=(
                temporal_confirmed
            ),
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.current_vision = (
            vision_status
        )

        self.current_vision_confidence = (
            vision_confidence
        )

        self.current_sensor_status = (
            sensor_status
        )

        self.current_sensor_score = (
            sensor_score
        )

        self.current_status = (
            final_status
        )

        self.current_reason = reason

        # ----------------------------------------------------
        # ALARM
        # ----------------------------------------------------

        self.alarm.update(
            final_status
        )

        # ----------------------------------------------------
        # EVENT
        # ----------------------------------------------------

        if (
            final_status
            != self.previous_status
        ):

            self.add_event(
                final_status,
                reason
            )

            self.previous_status = (
                final_status
            )

        # ----------------------------------------------------
        # UPDATE UI
        # ----------------------------------------------------

        self.update_status_ui(

            vision_status,
            vision_confidence,

            sensor_status,
            sensor_score,

            positive_frames,
            temporal_confirmed,

            final_status,
            final_confidence,

            reason,
        )

        # ----------------------------------------------------
        # DISPLAY CAMERA
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            annotated_frame,
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

        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(
            scaled
        )

    # ========================================================
    # STATUS UI
    # ========================================================

    def update_status_ui(

        self,

        vision_status=None,
        vision_confidence=None,

        sensor_status=None,
        sensor_score=None,

        positive_frames=None,
        temporal_confirmed=None,

        final_status=None,
        final_confidence=None,

        reason=None,
    ):

        if vision_status is None:
            vision_status = self.current_vision

        if vision_confidence is None:
            vision_confidence = (
                self.current_vision_confidence
            )

        if sensor_status is None:
            sensor_status = (
                self.current_sensor_status
            )

        if sensor_score is None:
            sensor_score = (
                self.current_sensor_score
            )

        if positive_frames is None:
            positive_frames = sum(
                self.temporal_buffer
            )

        if temporal_confirmed is None:
            temporal_confirmed = (
                positive_frames
                >= MIN_POSITIVE_FRAMES
            )

        if final_status is None:
            final_status = self.current_status

        if final_confidence is None:
            final_confidence = 1.0

        if reason is None:
            reason = self.current_reason

        # ----------------------------------------------------
        # STATUS COLOR
        # ----------------------------------------------------

        if final_status == "FIRE":

            status_color = RED

        elif final_status == "WARNING":

            status_color = AMBER

        else:

            status_color = GREEN

        # ----------------------------------------------------
        # MAIN STATUS
        # ----------------------------------------------------

        self.final_status_label.setText(
            final_status
        )

        self.final_status_label.setStyleSheet(
            f"""
            background-color: {status_color};
            color: #061016;
            border-radius: 8px;
            font-size: 24px;
            font-weight: 800;
            """
        )

        self.final_confidence_label.setText(
            f"Fusion Confidence  •  "
            f"{final_confidence * 100:.0f}%"
        )

        # ----------------------------------------------------
        # VISION
        # ----------------------------------------------------

        self.vision_value.setText(
            vision_status
        )

        self.vision_conf.setText(
            f"Fire: "
            f"{vision_confidence * 100:.0f}%"
        )

        if vision_status == "FIRE":

            self.vision_value.setStyleSheet(
                f"""
                color: {RED};
                font-size: 16px;
                font-weight: 700;
                """
            )

        elif vision_status == "SMOKE":

            self.vision_value.setStyleSheet(
                f"""
                color: {AMBER};
                font-size: 16px;
                font-weight: 700;
                """
            )

        else:

            self.vision_value.setStyleSheet(
                f"""
                color: {GREEN};
                font-size: 16px;
                font-weight: 700;
                """
            )

        # ----------------------------------------------------
        # SENSOR
        # ----------------------------------------------------

        self.sensor_value.setText(
            sensor_status
        )

        self.sensor_score.setText(
            f"Score: "
            f"{sensor_score * 100:.0f}%"
        )

        if sensor_status == "FIRE":

            self.sensor_value.setStyleSheet(
                f"""
                color: {RED};
                font-size: 16px;
                font-weight: 700;
                """
            )

        elif sensor_status == "WARNING":

            self.sensor_value.setStyleSheet(
                f"""
                color: {AMBER};
                font-size: 16px;
                font-weight: 700;
                """
            )

        else:

            self.sensor_value.setStyleSheet(
                f"""
                color: {GREEN};
                font-size: 16px;
                font-weight: 700;
                """
            )

        # ----------------------------------------------------
        # TEMPORAL
        # ----------------------------------------------------

        self.temporal_count.setText(
            f"{positive_frames} / {WINDOW_SIZE}"
        )

        if temporal_confirmed:

            self.temporal_state.setText(
                "● PERSISTENT EVIDENCE CONFIRMED"
            )

            self.temporal_state.setStyleSheet(
                f"""
                color: {RED};
                font-weight: 700;
                """
            )

        else:

            self.temporal_state.setText(
                "● Waiting for persistent evidence"
            )

            self.temporal_state.setStyleSheet(
                f"""
                color: {MUTED};
                """
            )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion_reason.setText(
            reason
        )

        if final_status == "FIRE":

            self.fusion_reason.setStyleSheet(
                f"""
                color: {RED};
                font-weight: 600;
                """
            )

        elif final_status == "WARNING":

            self.fusion_reason.setStyleSheet(
                f"""
                color: {AMBER};
                font-weight: 600;
                """
            )

        else:

            self.fusion_reason.setStyleSheet(
                f"""
                color: {MUTED};
                """
            )

    # ========================================================
    # EVENT LOG
    # ========================================================

    def add_event(
        self,
        status,
        reason
    ):

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        event = (
            f"[{timestamp}]  "
            f"{status}  —  {reason}"
        )

        self.events.insert(
            0,
            event
        )

        self.events = self.events[:5]

        self.event_label.setText(
            "\n".join(self.events)
        )

        self.event_count_label.setText(
            f"{len(self.events)} events"
        )

    # ========================================================
    # TEST ALARM
    # ========================================================

    def test_alarm(self):

        self.alarm.update(
            "FIRE"
        )

        self.add_event(
            "FIRE",
            "Manual fire alarm test activated"
        )

    # ========================================================
    # STOP ALARM
    # ========================================================

    def stop_alarm(self):

        self.alarm.stop()

        self.add_event(
            "NORMAL",
            "Alarm manually stopped"
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.alarm.stop()

        if self.cap:

            self.cap.release()

        cv2.destroyAllWindows()

        event.accept()


# ============================================================
# APPLICATION
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = SnapFireDashboard()

    window.show()

    sys.exit(
        app.exec()
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()