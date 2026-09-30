from dataclasses import dataclass


# ============================================================
# SNAPFIRE AI - SENSOR + VISION FUSION ENGINE
# ============================================================

@dataclass
class FusionResult:
    final_decision: str
    explanation: str
    sensor_prediction: str
    vision_prediction: str
    vision_confidence: float


def fuse_predictions(
    sensor_prediction,
    sensor_confidence,
    vision_prediction,
    vision_confidence
):
    """
    Combine sensor ML and camera vision predictions.

    Sensor model:
        Normal
        Warning
        Fire
        False Alarm

    Vision model:
        Fire
        Smoke
        None

    Confidence values:
        0.0 - 1.0
    """

    sensor_confidence = float(sensor_confidence)
    vision_confidence = float(vision_confidence)

    # ========================================================
    # 1. STRONG VISUAL FIRE
    # ========================================================

    if (
        vision_prediction == "Fire"
        and vision_confidence >= 0.60
    ):

        if sensor_prediction == "Fire":

            return FusionResult(
                "FIRE",
                "Sensor and vision both strongly confirm fire.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        elif sensor_prediction == "Warning":

            return FusionResult(
                "FIRE",
                "Strong visual fire detected with elevated sensor readings.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        elif sensor_prediction == "False Alarm":

            return FusionResult(
                "WARNING",
                "Strong visual fire detected, but sensor pattern indicates a possible false alarm. Verification required.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        else:

            return FusionResult(
                "WARNING",
                "Strong visual fire detected, but sensor evidence is currently normal.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

    # ========================================================
    # 2. MODERATE VISUAL FIRE
    # ========================================================

    if (
        vision_prediction == "Fire"
        and vision_confidence >= 0.30
    ):

        if sensor_prediction == "Fire":

            return FusionResult(
                "FIRE",
                "Sensor model confirms fire while vision detects possible fire.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        elif sensor_prediction == "Warning":

            return FusionResult(
                "WARNING",
                "Both sensor and vision indicate a possible fire condition.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        elif sensor_prediction == "False Alarm":

            return FusionResult(
                "FALSE ALARM",
                "Sensor pattern is consistent with a false alarm and visual fire confidence is moderate.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        else:

            return FusionResult(
                "WARNING",
                "Vision detects possible fire but sensor evidence is insufficient.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

    # ========================================================
    # 3. SMOKE DETECTION
    # ========================================================

    if vision_prediction == "Smoke":

        if sensor_prediction == "Fire":

            return FusionResult(
                "FIRE",
                "Sensor model indicates fire and smoke is detected visually.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        elif sensor_prediction == "Warning":

            return FusionResult(
                "WARNING",
                "Smoke detected visually with supporting sensor warning.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        else:

            return FusionResult(
                "WARNING",
                "Smoke detected by vision; sensor confirmation required.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

    # ========================================================
    # 4. SENSOR-ONLY FIRE
    # ========================================================

    if sensor_prediction == "Fire":

        if sensor_confidence >= 0.80:

            return FusionResult(
                "FIRE",
                "High-confidence sensor model indicates fire; visual confirmation is currently unavailable.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

        else:

            return FusionResult(
                "WARNING",
                "Sensor model indicates possible fire, but confidence is below the fire threshold.",
                sensor_prediction,
                vision_prediction,
                vision_confidence
            )

    # ========================================================
    # 5. SENSOR WARNING
    # ========================================================

    if sensor_prediction == "Warning":

        return FusionResult(
            "WARNING",
            "Sensor readings indicate an abnormal condition.",
            sensor_prediction,
            vision_prediction,
            vision_confidence
        )

    # ========================================================
    # 6. FALSE ALARM
    # ========================================================

    if sensor_prediction == "False Alarm":

        return FusionResult(
            "FALSE ALARM",
            "Sensor pattern is consistent with a false alarm.",
            sensor_prediction,
            vision_prediction,
            vision_confidence
        )

    # ========================================================
    # 7. NORMAL
    # ========================================================

    return FusionResult(
        "NORMAL",
        "No significant fire or smoke evidence detected.",
        sensor_prediction,
        vision_prediction,
        vision_confidence
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       SNAPFIRE AI - FUSION ENGINE")
    print("=" * 60)

    test_cases = [

        # 1. Completely normal
        ("Normal", 0.93, "None", 0.00),

        # 2. Sensor warning + moderate vision fire
        ("Warning", 0.88, "Fire", 0.35),

        # 3. Both confirm fire
        ("Fire", 1.00, "Fire", 0.80),

        # 4. False alarm + weak visual fire
        ("False Alarm", 0.95, "Fire", 0.28),

        # 5. Normal sensor + smoke
        ("Normal", 0.93, "Smoke", 0.45),

        # 6. HIGH CONFIDENCE SENSOR FIRE + NO VISION
        ("Fire", 1.00, "None", 0.00),

        # 7. HIGH CONFIDENCE SENSOR FIRE + MODERATE VISION
        ("Fire", 1.00, "Fire", 0.35),

        # 8. WARNING SENSOR + STRONG VISION
        ("Warning", 0.88, "Fire", 0.75),

        # 9. NORMAL SENSOR + MODERATE VISION FIRE
        ("Normal", 0.93, "Fire", 0.40),

    ]

    for (
        sensor_prediction,
        sensor_confidence,
        vision_prediction,
        vision_confidence
    ) in test_cases:

        result = fuse_predictions(
            sensor_prediction,
            sensor_confidence,
            vision_prediction,
            vision_confidence
        )

        print("\n----------------------------------------")

        print(
            f"Sensor prediction : {sensor_prediction}"
        )

        print(
            f"Sensor confidence : {sensor_confidence * 100:.1f}%"
        )

        print(
            f"Vision prediction : {vision_prediction}"
        )

        print(
            f"Vision confidence : {vision_confidence * 100:.1f}%"
        )

        print(
            f"FINAL DECISION    : {result.final_decision}"
        )

        print(
            f"Explanation       : {result.explanation}"
        )

    print("\n" + "=" * 60)
    print("       FUSION ENGINE TEST COMPLETE")
    print("=" * 60)