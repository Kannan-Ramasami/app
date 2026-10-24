"""
SignSpeak Personal Data Collection & Calibration Tool
Interactive tool to record personal ISL hand gestures for all 26 letters,
strictly aligned with the ISL Reference Chart.
"""

import os
import sys
import json
import time
import cv2
import numpy as np

try:
    from backend.hand_detector import HandDetector
    from backend.signs_data import compute_normalized_features, distance
    from backend.train_model import train_all
except ImportError:
    from hand_detector import HandDetector
    from signs_data import compute_normalized_features, distance
    from train_model import train_all

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
COLLECTED_FILE = os.path.join(DATA_DIR, "collected_samples.json")

# Complete 26 ISL Alphabet strictly aligned with reference chart
ISL_ALPHABET = [
    {"letter": "A", "type": "two_hand", "desc": "Both thumbs up touching each other", "hands": "Two Hands"},
    {"letter": "B", "type": "two_hand", "desc": "Both hands forming 'O' touching side-by-side", "hands": "Two Hands"},
    {"letter": "C", "type": "one_hand", "desc": "One hand curved in 'C' shape", "hands": "One Hand"},
    {"letter": "D", "type": "two_hand", "desc": "Left index up, right curved hand touches to form 'D'", "hands": "Two Hands"},
    {"letter": "E", "type": "two_hand", "desc": "Left index up, right index touches left index tip", "hands": "Two Hands"},
    {"letter": "F", "type": "two_hand", "desc": "Left index+middle extended, right index+middle across", "hands": "Two Hands"},
    {"letter": "G", "type": "two_hand", "desc": "Both fists stacked vertically (one on top of other)", "hands": "Two Hands"},
    {"letter": "H", "type": "two_hand", "desc": "Left flat open palm upright, right palm sweeps across", "hands": "Two Hands"},
    {"letter": "I", "type": "one_hand", "desc": "Index finger pointing straight up, other fingers in fist", "hands": "One Hand"},
    {"letter": "J", "type": "one_hand", "desc": "One hand with index hooked/curved down in 'J' shape", "hands": "One Hand"},
    {"letter": "K", "type": "two_hand", "desc": "Left index up, right hooked index touches knuckle for 'K'", "hands": "Two Hands"},
    {"letter": "L", "type": "one_hand", "desc": "Thumb and index extended at 90 deg forming 'L'", "hands": "One Hand"},
    {"letter": "M", "type": "two_hand", "desc": "Left open palm down, right 3 fingers on left palm", "hands": "Two Hands"},
    {"letter": "N", "type": "two_hand", "desc": "Left open palm down, right 2 fingers on left palm", "hands": "Two Hands"},
    {"letter": "O", "type": "one_hand", "desc": "Thumb and fingers touching in an 'O' circle", "hands": "One Hand"},
    {"letter": "P", "type": "two_hand", "desc": "Left index up, right thumb+index forms 'P' loop at top", "hands": "Two Hands"},
    {"letter": "Q", "type": "two_hand", "desc": "Left hand 'O' circle, right index hooked through circle", "hands": "Two Hands"},
    {"letter": "R", "type": "two_hand", "desc": "Left open palm, right curved index resting on palm", "hands": "Two Hands"},
    {"letter": "S", "type": "two_hand", "desc": "Both hands in fists with pinky fingers hooked together", "hands": "Two Hands"},
    {"letter": "T", "type": "two_hand", "desc": "Left index horizontal, right index pointing up under it", "hands": "Two Hands"},
    {"letter": "U", "type": "one_hand", "desc": "Thumb and pinky upright in 'U', other 3 fingers curled", "hands": "One Hand"},
    {"letter": "V", "type": "one_hand", "desc": "Index and middle extended in 'V' (peace sign)", "hands": "One Hand"},
    {"letter": "W", "type": "one_hand", "desc": "3 fingers (index, middle, ring) upright in 'W' shape", "hands": "One Hand"},
    {"letter": "X", "type": "two_hand", "desc": "Both index fingers crossed diagonally in an 'X'", "hands": "Two Hands"},
    {"letter": "Y", "type": "two_hand", "desc": "Left thumb+index in 'V', right index slicing between", "hands": "Two Hands"},
    {"letter": "Z", "type": "two_hand", "desc": "Left flat palm, right fingertips bent at 90 deg on palm", "hands": "Two Hands"}
]

SAMPLES_PER_LETTER = 35


def load_existing():
    if os.path.exists(COLLECTED_FILE):
        try:
            with open(COLLECTED_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_data(data):
    with open(COLLECTED_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved progress to {COLLECTED_FILE}")


def main():
    collected_data = load_existing()
    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    current_idx = 0
    for i, item in enumerate(ISL_ALPHABET):
        if item["letter"] not in collected_data:
            current_idx = i
            break

    state = "IDLE"
    countdown_start = 0.0
    countdown_duration = 3.0
    recorded_samples = []
    done_start_time = 0.0

    print("=" * 60)
    print("SignSpeak ISL Personal Calibration Tool")
    print("Controls:")
    print("  [SPACE] : Start 3s countdown / confirm next")
    print("  [R]     : Re-record current letter")
    print("  [S]     : Skip to next letter")
    print("  [P]     : Go back to previous letter")
    print("  [T]     : Train model immediately with current data")
    print("  [Q]     : Quit and save")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        detections = detector.process_frame(frame)
        detector.draw_landmarks(frame, detections)

        num_hands = len(detections)
        now = time.time()
        current_item = ISL_ALPHABET[current_idx]
        letter = current_item["letter"]
        letter_type = current_item["type"]
        desc = current_item["desc"]
        hands_req = current_item["hands"]

        hands_ready = (num_hands >= 2) if letter_type == "two_hand" else (num_hands >= 1)

        # State Machine
        if state == "COUNTDOWN":
            elapsed = now - countdown_start
            remaining = countdown_duration - elapsed
            if remaining <= 0:
                state = "RECORDING"
                recorded_samples = []
            else:
                cd_num = int(np.ceil(remaining))
                cv2.circle(frame, (w // 2, h // 2), 90, (20, 20, 20), -1)
                cv2.circle(frame, (w // 2, h // 2), 90, (0, 200, 255), 5)
                cv2.putText(
                    frame,
                    str(cd_num),
                    (w // 2 - 25, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3.0,
                    (0, 255, 255),
                    6
                )

        elif state == "RECORDING":
            if letter_type == "two_hand" and num_hands >= 2:
                pts1 = detections[0]["points"]
                pts2 = detections[1]["points"]
                f1 = compute_normalized_features(pts1)
                f2 = compute_normalized_features(pts2)
                palm_ref = max(10.0, distance(pts1[0], pts1[9]))
                dy = abs(pts1[0][1] - pts2[0][1]) / palm_ref
                recorded_samples.append({
                    "hand1": f1,
                    "hand2": f2,
                    "dy": dy
                })
            elif letter_type == "one_hand" and num_hands >= 1:
                pts = detections[0]["points"]
                f = compute_normalized_features(pts)
                recorded_samples.append({
                    "hand": f
                })

            if len(recorded_samples) >= SAMPLES_PER_LETTER:
                collected_data[letter] = {
                    "letter": letter,
                    "hand_type": letter_type,
                    "samples": recorded_samples
                }
                save_data(collected_data)
                state = "DONE_LETTER"
                done_start_time = now

        elif state == "DONE_LETTER":
            if now - done_start_time > 1.8:
                if current_idx < len(ISL_ALPHABET) - 1:
                    current_idx += 1
                    state = "IDLE"
                else:
                    state = "ALL_DONE"

        # -----------------------------
        # DRAW TOP HUD CARD
        # -----------------------------
        cv2.rectangle(frame, (20, 20), (w - 20, 140), (22, 22, 25), -1)
        border_color = (0, 230, 100) if letter in collected_data else (60, 60, 65)
        cv2.rectangle(frame, (20, 20), (w - 20, 140), border_color, 2)

        badge_bg = (0, 180, 80) if letter in collected_data else (50, 50, 60)
        cv2.rectangle(frame, (35, 35), (135, 125), badge_bg, -1)
        cv2.putText(
            frame,
            letter,
            (55, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (255, 255, 255),
            4
        )

        rec_status = "SAVED" if letter in collected_data else "NOT RECORDED"
        rec_color = (0, 230, 100) if letter in collected_data else (120, 120, 120)
        cv2.putText(
            frame,
            f"Sign: {letter} ({hands_req})  [{rec_status}]",
            (155, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Hint: {desc}",
            (155, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        cv2.putText(
            frame,
            f"Progress: {current_idx + 1}/{len(ISL_ALPHABET)} letters  ({len(collected_data)} total recorded)",
            (155, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            rec_color,
            1
        )

        # -----------------------------
        # STATUS / PROMPT BAR
        # -----------------------------
        if state == "IDLE":
            if hands_ready:
                msg = f"Hands detected! Press [SPACE] to record {letter}"
                msg_color = (0, 255, 150)
            else:
                req_text = "both hands" if letter_type == "two_hand" else "one hand"
                msg = f"Please show {req_text} in camera frame..."
                msg_color = (0, 180, 255)

            cv2.rectangle(frame, (20, h - 90), (w - 20, h - 30), (20, 20, 25), -1)
            cv2.putText(frame, msg, (40, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

        elif state == "RECORDING":
            progress = len(recorded_samples) / SAMPLES_PER_LETTER
            bar_w = int((w - 80) * progress)

            cv2.rectangle(frame, (20, h - 90), (w - 20, h - 30), (20, 20, 25), -1)
            cv2.rectangle(frame, (40, h - 75), (40 + bar_w, h - 45), (0, 230, 100), -1)
            cv2.rectangle(frame, (40, h - 75), (w - 40, h - 45), (100, 100, 100), 1)
            cv2.putText(
                frame,
                f"HOLD GESTURE... Recording {len(recorded_samples)}/{SAMPLES_PER_LETTER}",
                (50, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        elif state == "DONE_LETTER":
            cv2.rectangle(frame, (20, h - 90), (w - 20, h - 30), (20, 40, 20), -1)
            cv2.putText(
                frame,
                f"Letter '{letter}' recorded successfully! Advancing...",
                (40, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 100),
                2
            )

        elif state == "ALL_DONE":
            cv2.rectangle(frame, (40, h // 2 - 80), (w - 40, h // 2 + 80), (20, 25, 20), -1)
            cv2.rectangle(frame, (40, h // 2 - 80), (w - 40, h // 2 + 80), (0, 255, 100), 2)
            cv2.putText(
                frame,
                "ALL 26 LETTERS RECORDED!",
                (w // 2 - 220, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 120),
                3
            )
            cv2.putText(
                frame,
                "Press [T] to train Random Forest model now, or [Q] to exit.",
                (w // 2 - 260, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "[SPACE]: Record  [R]: Retake  [S]: Skip  [P]: Prev  [T]: Train Model  [Q]: Quit",
            (30, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1
        )

        cv2.imshow("SignSpeak ISL Personal Calibration Tool", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
            elif state == "DONE_LETTER":
                if current_idx < len(ISL_ALPHABET) - 1:
                    current_idx += 1
                    state = "IDLE"
        elif key == ord("r"):
            state = "COUNTDOWN"
            countdown_start = time.time()
        elif key == ord("s"):
            if current_idx < len(ISL_ALPHABET) - 1:
                current_idx += 1
                state = "IDLE"
        elif key == ord("p"):
            if current_idx > 0:
                current_idx -= 1
                state = "IDLE"
        elif key == ord("t"):
            print("Starting model retraining with personal calibration data...")
            cv2.destroyAllWindows()
            train_all()
            print("Retraining complete! You can now run check_cord.py to test recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
