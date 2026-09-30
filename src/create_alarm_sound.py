import pygame
import math
import os
import wave
import struct


OUTPUT_DIR = r"F:\SnapFire-AI\assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_tone(filename, frequency, duration, volume=0.5):
    sample_rate = 44100
    num_samples = int(sample_rate * duration)

    with wave.open(filename, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(num_samples):
            t = i / sample_rate

            # Smooth tone
            sample = math.sin(2 * math.pi * frequency * t)

            # Small fade in/out
            fade = min(
                1.0,
                i / (sample_rate * 0.01),
                (num_samples - i) / (sample_rate * 0.01)
            )

            value = int(32767 * volume * sample * fade)

            wav.writeframes(struct.pack("<h", value))


def generate_warning():
    filename = os.path.join(OUTPUT_DIR, "warning.wav")

    sample_rate = 44100
    beep_duration = 0.25
    pause_duration = 0.45

    with wave.open(filename, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for _ in range(4):

            # Beep
            samples = int(sample_rate * beep_duration)

            for i in range(samples):
                t = i / sample_rate
                sample = math.sin(2 * math.pi * 900 * t)

                fade = min(
                    1.0,
                    i / (sample_rate * 0.01),
                    (samples - i) / (sample_rate * 0.01)
                )

                value = int(32767 * 0.5 * sample * fade)
                wav.writeframes(struct.pack("<h", value))

            # Pause
            pause_samples = int(sample_rate * pause_duration)

            for _ in range(pause_samples):
                wav.writeframes(struct.pack("<h", 0))

    print("Created:", filename)


def generate_fire_alarm():
    filename = os.path.join(OUTPUT_DIR, "fire_alarm.wav")

    sample_rate = 44100

    with wave.open(filename, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        # Rapid alternating alarm tones
        for _ in range(8):

            for frequency in [900, 1400]:

                duration = 0.18
                samples = int(sample_rate * duration)

                for i in range(samples):
                    t = i / sample_rate
                    sample = math.sin(2 * math.pi * frequency * t)

                    fade = min(
                        1.0,
                        i / (sample_rate * 0.01),
                        (samples - i) / (sample_rate * 0.01)
                    )

                    value = int(32767 * 0.65 * sample * fade)
                    wav.writeframes(struct.pack("<h", value))

    print("Created:", filename)


if __name__ == "__main__":

    print("Generating SnapFire AI alarm sounds...")

    generate_warning()
    generate_fire_alarm()

    print()
    print("Alarm sounds created successfully.")
    print("Location:", OUTPUT_DIR)