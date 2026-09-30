"""
SnapFire AI - Sensor Evidence Layer

Prototype rule-based sensor interpretation.

This layer converts physical sensor readings into:
    NORMAL
    WARNING
    FIRE

The thresholds are prototype values and are NOT safety-certified.
"""

def evaluate_sensors(
    temperature,
    humidity,
    tvoc,
    eco2,
    pm1,
    pm25,
    nc05,
    nc10,
    nc25
):

    evidence = []
    score = 0.0

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if temperature >= 45:
        score += 0.30
        evidence.append("High temperature")

    elif temperature >= 35:
        score += 0.15
        evidence.append("Elevated temperature")


    # --------------------------------------------------------
    # eCO2
    # --------------------------------------------------------

    if eco2 >= 5000:
        score += 0.25
        evidence.append("Very high eCO2")

    elif eco2 >= 2000:
        score += 0.12
        evidence.append("Elevated eCO2")


    # --------------------------------------------------------
    # TVOC
    # --------------------------------------------------------

    if tvoc >= 3000:
        score += 0.20
        evidence.append("High TVOC")

    elif tvoc >= 1000:
        score += 0.10
        evidence.append("Elevated TVOC")


    # --------------------------------------------------------
    # PARTICULATE MATTER
    # --------------------------------------------------------

    if pm25 >= 1000:
        score += 0.35
        evidence.append("Very high PM2.5")

    elif pm25 >= 100:
        score += 0.15
        evidence.append("Elevated PM2.5")


    if pm1 >= 500:
        score += 0.20
        evidence.append("Very high PM1.0")

    elif pm1 >= 100:
        score += 0.10
        evidence.append("Elevated PM1.0")


    # --------------------------------------------------------
    # NUMBER CONCENTRATION
    # --------------------------------------------------------

    if nc05 >= 1000:
        score += 0.15
        evidence.append("High NC0.5")

    if nc10 >= 1000:
        score += 0.15
        evidence.append("High NC1.0")

    if nc25 >= 500:
        score += 0.15
        evidence.append("High NC2.5")


    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    if humidity < 20:
        score += 0.05
        evidence.append("Low humidity")


    # --------------------------------------------------------
    # LIMIT SCORE
    # --------------------------------------------------------

    score = min(score, 1.0)


    # --------------------------------------------------------
    # FINAL SENSOR STATUS
    # --------------------------------------------------------

    if score >= 0.65:

        status = "FIRE"

    elif score >= 0.25:

        status = "WARNING"

    else:

        status = "NORMAL"


    # --------------------------------------------------------
    # DEFAULT MESSAGE
    # --------------------------------------------------------

    if not evidence:
        evidence.append(
            "No significant sensor anomaly"
        )


    return {
        "status": status,
        "confidence": round(score, 3),
        "evidence": evidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    scenarios = {

        "Normal": {
            "temperature": 25,
            "humidity": 50,
            "tvoc": 100,
            "eco2": 450,
            "pm1": 2,
            "pm25": 3,
            "nc05": 10,
            "nc10": 2,
            "nc25": 1
        },

        "Warning": {
            "temperature": 36,
            "humidity": 35,
            "tvoc": 1200,
            "eco2": 2200,
            "pm1": 120,
            "pm25": 150,
            "nc05": 150,
            "nc10": 100,
            "nc25": 50
        },

        "Fire": {
            "temperature": 50,
            "humidity": 18,
            "tvoc": 3500,
            "eco2": 6000,
            "pm1": 1000,
            "pm25": 2000,
            "nc05": 5000,
            "nc10": 3000,
            "nc25": 1500
        }
    }


    print("=" * 70)
    print("SNAPFIRE AI - SENSOR EVIDENCE TEST")
    print("=" * 70)

    for name, values in scenarios.items():

        result = evaluate_sensors(**values)

        print("\n" + "-" * 70)
        print(name)

        print(
            "Status     :",
            result["status"]
        )

        print(
            "Confidence :",
            result["confidence"]
        )

        print(
            "Evidence   :",
            ", ".join(result["evidence"])
        )