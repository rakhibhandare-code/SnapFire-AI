"""
SnapFire AI - Multimodal Fusion Engine V2

Combines:
    1. Vision evidence from YOLO
    2. Sensor evidence
    3. Temporal confirmation

Final states:
    NORMAL
    WARNING
    FIRE
    FALSE ALARM

This is a prototype decision layer and is NOT a safety-certified
fire alarm controller.
"""


# ============================================================
# CONFIGURATION
# ============================================================

VISION_FIRE_THRESHOLD = 0.25
VISION_SMOKE_THRESHOLD = 0.25

STRONG_VISION_FIRE = 0.60
STRONG_SENSOR_FIRE = 0.80


# ============================================================
# MAIN FUSION FUNCTION
# ============================================================

def fuse_evidence(
    vision_status="NORMAL",
    fire_confidence=0.0,
    smoke_confidence=0.0,
    sensor_status="NORMAL",
    sensor_confidence=0.0,
    vision_confirmed=False
):

    # --------------------------------------------------------
    # Normalize inputs
    # --------------------------------------------------------

    vision_status = str(
        vision_status
    ).upper()

    sensor_status = str(
        sensor_status
    ).upper()

    fire_confidence = max(
        0.0,
        min(1.0, float(fire_confidence))
    )

    smoke_confidence = max(
        0.0,
        min(1.0, float(smoke_confidence))
    )

    sensor_confidence = max(
        0.0,
        min(1.0, float(sensor_confidence))
    )


    # ========================================================
    # DETERMINE VISION EVIDENCE
    # ========================================================

    vision_fire = (
        vision_status == "FIRE"
        or fire_confidence >= VISION_FIRE_THRESHOLD
    )

    vision_smoke = (
        vision_status == "SMOKE"
        or smoke_confidence >= VISION_SMOKE_THRESHOLD
    )


    # ========================================================
    # CASE 1
    # STRONG AGREEMENT
    # ========================================================

    if (
        vision_fire
        and vision_confirmed
        and sensor_status == "FIRE"
    ):

        return {
            "status": "FIRE",
            "reason": "Vision and sensor evidence agree",
            "confidence": calculate_confidence(
                fire_confidence,
                sensor_confidence
            )
        }


    # ========================================================
    # CASE 2
    # STRONG SENSOR + CONFIRMED VISION
    # ========================================================

    if (
        vision_fire
        and vision_confirmed
        and sensor_confidence >= STRONG_SENSOR_FIRE
    ):

        return {
            "status": "FIRE",
            "reason": "Confirmed vision with strong sensor evidence",
            "confidence": calculate_confidence(
                fire_confidence,
                sensor_confidence
            )
        }


    # ========================================================
    # CASE 3
    # STRONG CONFIRMED VISION
    #
    # Useful when physical sensor data is unavailable.
    # ========================================================

    if (
        vision_confirmed
        and fire_confidence >= STRONG_VISION_FIRE
    ):

        return {
            "status": "FIRE",
            "reason": "Persistent strong visual fire evidence",
            "confidence": fire_confidence
        }


    # ========================================================
    # CASE 4
    # SENSOR FIRE WITHOUT VISION
    # ========================================================

    if (
        sensor_status == "FIRE"
        and sensor_confidence >= STRONG_SENSOR_FIRE
    ):

        return {
            "status": "WARNING",
            "reason": "Strong sensor evidence without visual confirmation",
            "confidence": sensor_confidence
        }


    # ========================================================
    # CASE 5
    # VISION FIRE BUT SENSOR CONTRADICTS IT
    #
    # This is exactly where multimodal fusion helps reduce
    # isolated visual false alarms.
    # ========================================================

    if (
        vision_fire
        and sensor_status == "NORMAL"
    ):

        return {
            "status": "WARNING",
            "reason": "Visual fire indication but sensors are normal",
            "confidence": fire_confidence
        }


    # ========================================================
    # CASE 6
    # SMOKE DETECTION
    # ========================================================

    if vision_smoke:

        if sensor_status in ["WARNING", "FIRE"]:

            return {
                "status": "WARNING",
                "reason": "Visual smoke with sensor anomaly",
                "confidence": max(
                    smoke_confidence,
                    sensor_confidence
                )
            }

        return {
            "status": "WARNING",
            "reason": "Visual smoke detected",
            "confidence": smoke_confidence
        }


    # ========================================================
    # CASE 7
    # SENSOR WARNING
    # ========================================================

    if sensor_status == "WARNING":

        return {
            "status": "WARNING",
            "reason": "Sensor anomaly detected",
            "confidence": sensor_confidence
        }


    # ========================================================
    # CASE 8
    # POSSIBLE FALSE ALARM
    #
    # A visual indication existed but did not persist.
    # ========================================================

    if (
        fire_confidence >= VISION_FIRE_THRESHOLD
        and not vision_confirmed
    ):

        return {
            "status": "FALSE ALARM",
            "reason": "Visual indication not temporally confirmed",
            "confidence": fire_confidence
        }


    # ========================================================
    # DEFAULT
    # ========================================================

    return {
        "status": "NORMAL",
        "reason": "No significant fire or smoke evidence",
        "confidence": 1.0
    }


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(
    vision_confidence,
    sensor_confidence
):

    # Weighted combination.
    # This is a prototype fusion score, NOT a probability.

    score = (
        0.55 * vision_confidence
        +
        0.45 * sensor_confidence
    )

    return round(
        min(score, 1.0),
        3
    )


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    tests = [

        {
            "name": "Normal",
            "vision_status": "NORMAL",
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0,
            "sensor_status": "NORMAL",
            "sensor_confidence": 0.05,
            "vision_confirmed": False
        },

        {
            "name": "Single Visual Detection",
            "vision_status": "FIRE",
            "fire_confidence": 0.35,
            "smoke_confidence": 0.0,
            "sensor_status": "NORMAL",
            "sensor_confidence": 0.10,
            "vision_confirmed": False
        },

        {
            "name": "Persistent Visual Fire",
            "vision_status": "FIRE",
            "fire_confidence": 0.72,
            "smoke_confidence": 0.0,
            "sensor_status": "NORMAL",
            "sensor_confidence": 0.20,
            "vision_confirmed": True
        },

        {
            "name": "Vision + Sensor Fire",
            "vision_status": "FIRE",
            "fire_confidence": 0.75,
            "smoke_confidence": 0.0,
            "sensor_status": "FIRE",
            "sensor_confidence": 0.90,
            "vision_confirmed": True
        },

        {
            "name": "Smoke + Sensor Warning",
            "vision_status": "SMOKE",
            "fire_confidence": 0.0,
            "smoke_confidence": 0.55,
            "sensor_status": "WARNING",
            "sensor_confidence": 0.65,
            "vision_confirmed": True
        },

        {
            "name": "Sensor Fire Without Vision",
            "vision_status": "NORMAL",
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0,
            "sensor_status": "FIRE",
            "sensor_confidence": 0.92,
            "vision_confirmed": False
        }
    ]


    print("=" * 70)
    print("SNAPFIRE AI - FUSION ENGINE V2 TEST")
    print("=" * 70)

    for test in tests:

        name = test.pop("name")

        result = fuse_evidence(
            **test
        )

        print("\n" + "-" * 70)
        print(name)

        print(
            "Status     :",
            result["status"]
        )

        print(
            "Reason     :",
            result["reason"]
        )

        print(
            "Confidence :",
            result["confidence"]
        )