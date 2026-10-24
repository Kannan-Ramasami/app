import os
import json
import cv2
import numpy as np
import time
from collections import deque, Counter

try:
    from backend.signs_data import (
        SIGNS, ONE_HAND_LETTERS, TWO_HAND_LETTERS,
        distance, angle, compute_normalized_features, validate_one_hand
    )
    from backend.classifier import SignClassifier
    from backend.hand_detector import HandDetector
    from backend.train_model import train_all, COLLECTED_SAMPLES_FILE
except ImportError:
    from signs_data import (
        SIGNS, ONE_HAND_LETTERS, TWO_HAND_LETTERS,
        distance, angle, compute_normalized_features, validate_one_hand
    )
    from classifier import SignClassifier
    from hand_detector import HandDetector
    from train_model import train_all, COLLECTED_SAMPLES_FILE


def save_calibrated_batch(letter: str, batch_vecs: list[list[float]]) -> None:
    """Replaces the calibrated samples for `letter` with a clean, steady 30-frame batch."""
    data: dict[str, list[list[float]]] = {}
    if os.path.exists(COLLECTED_SAMPLES_FILE):
        try:
            with open(COLLECTED_SAMPLES_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[letter] = [[round(float(x), 4) for x in vec] for vec in batch_vecs]
    os.makedirs(os.path.dirname(COLLECTED_SAMPLES_FILE), exist_ok=True)
    with open(COLLECTED_SAMPLES_FILE, "w") as f:
        json.dump(data, f)


def run_recognition() -> None:
    print("=" * 65)
    print("SignSpeak ISL Recognition | High-Stability 26-Letter Model")
    print("Controls in Camera Window:")
    print("  [c] -> Press 'c' then any letter (A-Z): 1.5s countdown -> records & auto-retrains!")
    print("  [t] -> Manual retrain & hot-reload")
    print("  [q] -> Quit")
    print("=" * 65)

    try:
        classifier = SignClassifier()
        print("ML Ensemble models loaded successfully.")
    except Exception as e:
        print(f"Models not found ({e}). Running initial training now...")
        train_all()
        classifier = SignClassifier()

    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()
    fps = 0.0

    history_len = 8
    pred_history: deque[str | None] = deque(maxlen=history_len)
    conf_history: deque[float] = deque(maxlen=history_len)
    last_two_hand_time = 0.0

    awaiting_letter_key = False
    calibrating_letter: str | None = None
    calib_countdown_until = 0.0
    calib_frames_left = 0
    calib_buffer: list[list[float]] = []
    CALIB_TARGET_FRAMES = 30
    status_toast = ""
    status_toast_until = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        detections = detector.process_frame(frame)
        detector.draw_landmarks(frame, detections)
        num_hands = len(detections)

        raw_letter: str | None = None
        raw_conf = 0.0
        top_candidates: list[tuple[str, float]] = []
        current_two_vec: list[float] | None = None
        current_one_vec: list[float] | None = None

        if num_hands >= 2:
            last_two_hand_time = curr_time
            pts1 = detections[0]["points"]
            pts2 = detections[1]["points"]

            feat1 = compute_normalized_features(pts1)
            feat2 = compute_normalized_features(pts2)

            palm_ref = max(10.0, distance(pts1[0], pts1[9]), distance(pts2[0], pts2[9]))
            dy = abs(pts1[0][1] - pts2[0][1]) / palm_ref
            current_two_vec = list(feat1) + list(feat2) + [dy]

            raw_letter, raw_conf, top_candidates = classifier.predict(
                hand1=feat1,
                hand2=feat2,
                dy=dy,
                min_confidence=0.36,
                return_top_k=True,
                pts1=pts1,
                pts2=pts2,
            )
        elif num_hands == 1:
            pts = detections[0]["points"]
            feat = compute_normalized_features(pts)
            current_one_vec = list(feat)

            if curr_time - last_two_hand_time < 0.65:
                raw_letter = None
                raw_conf = 0.0
            else:
                raw_letter, raw_conf, top_candidates = classifier.predict(
                    hand1=feat,
                    hand2=None,
                    min_confidence=0.45,
                    return_top_k=True,
                    pts1=pts,
                )

        # Handle live calibration with 1.5s preparation countdown
        if calibrating_letter is not None and curr_time >= calib_countdown_until and calib_frames_left > 0:
            if calibrating_letter in TWO_HAND_LETTERS and current_two_vec is not None:
                calib_buffer.append(current_two_vec)
                calib_frames_left -= 1
            elif calibrating_letter in ONE_HAND_LETTERS and current_one_vec is not None:
                calib_buffer.append(current_one_vec)
                calib_frames_left -= 1

            if calib_frames_left == 0:
                save_calibrated_batch(calibrating_letter, calib_buffer)
                print(f"Saved 30 clean frames for '{calibrating_letter}'. Auto-retraining now...")
                train_all()
                classifier.reload_models()
                status_toast = f"'{calibrating_letter}' Calibrated & Model Reloaded!"
                status_toast_until = time.time() + 4.5
                calibrating_letter = None
                calib_buffer = []

        if raw_letter is not None:
            pred_history.append(raw_letter)
            conf_history.append(raw_conf)
        else:
            pred_history.append(None)
            conf_history.append(0.0)

        active_preds = [p for p in pred_history if p is not None]
        smoothed_letter: str | None = None
        smoothed_conf = 0.0

        if len(active_preds) >= 4:
            counts = Counter(active_preds)
            most_common_letter, count = counts.most_common(1)[0]
            if count >= 4:
                smoothed_letter = most_common_letter
                letter_confs = [c for p, c in zip(pred_history, conf_history) if p == most_common_letter]
                smoothed_conf = float(np.mean(letter_confs)) if letter_confs else 0.0

        # -----------------------------
        # DRAW HUD INTERFACE
        # -----------------------------
        cv2.rectangle(frame, (0, 0), (w, 50), (18, 18, 22), -1)
        cv2.putText(
            frame,
            "SignSpeak ISL | [C] Calibrate Sign  [T] Retrain  [Q] Quit",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        h1_color = (0, 230, 100) if num_hands >= 1 else (70, 70, 75)
        h2_color = (0, 230, 100) if num_hands >= 2 else (70, 70, 75)
        cv2.putText(frame, "Hand 1 [OK]" if num_hands >= 1 else "Hand 1 [--]", (w - 250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h1_color, 2)
        cv2.putText(frame, "Hand 2 [OK]" if num_hands >= 2 else "Hand 2 [--]", (w - 125, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h2_color, 2)

        if smoothed_letter is not None:
            pct = int(smoothed_conf * 100)
            is_two_hand = smoothed_letter in TWO_HAND_LETTERS and num_hands >= 2
            type_text = "Two Hands" if is_two_hand else "One Hand"

            cv2.rectangle(frame, (30, 70), (280, 210), (18, 18, 22), -1)
            cv2.rectangle(frame, (30, 70), (280, 210), (0, 230, 100), 2)
            cv2.putText(frame, smoothed_letter, (60, 165), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 120), 4)
            cv2.putText(frame, f"{type_text} | {pct}%", (55, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.rectangle(frame, (30, 70), (280, 135), (25, 25, 28), -1)
            cv2.putText(frame, "Show ISL Sign...", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

        if top_candidates:
            cand_str = "Top Matches: " + "  |  ".join([f"{l}: {int(c*100)}%" for l, c in top_candidates])
            cv2.rectangle(frame, (20, h - 55), (480, h - 15), (18, 18, 22), -1)
            cv2.putText(frame, cand_str, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 1)

        if awaiting_letter_key:
            cv2.rectangle(frame, (30, 225), (540, 265), (0, 140, 255), -1)
            cv2.putText(frame, "PRESS LETTER KEY (A-Z) TO CALIBRATE...", (42, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif calibrating_letter is not None:
            if curr_time < calib_countdown_until:
                rem = calib_countdown_until - curr_time
                cv2.rectangle(frame, (30, 225), (560, 265), (0, 165, 255), -1)
                cv2.putText(
                    frame,
                    f"HOLD SIGN '{calibrating_letter}' STEADY... Starting in {rem:.1f}s",
                    (42, 252),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2
                )
            else:
                req = 2 if calibrating_letter in TWO_HAND_LETTERS else 1
                cv2.rectangle(frame, (30, 225), (560, 265), (0, 180, 80), -1)
                cv2.putText(
                    frame,
                    f"RECORDING '{calibrating_letter}' ({req} Hand): {calib_frames_left} frames left...",
                    (42, 252),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2
                )
        elif curr_time < status_toast_until:
            cv2.rectangle(frame, (30, 225), (560, 265), (35, 35, 45), -1)
            cv2.putText(frame, status_toast, (42, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (w - 110, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("SignSpeak ISL Recognition (Enhanced Ensemble ML)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        if awaiting_letter_key:
            char = chr(key).upper()
            if "A" <= char <= "Z":
                calibrating_letter = char
                calib_frames_left = CALIB_TARGET_FRAMES
                calib_buffer = []
                calib_countdown_until = time.time() + 1.5
                print(f"Get ready to hold '{char}' (recording starts in 1.5s)...")
            awaiting_letter_key = False
        else:
            if key == ord("q"):
                break
            elif key == ord("c"):
                awaiting_letter_key = True
            elif key == ord("t"):
                train_all()
                classifier.reload_models()
                status_toast = "Models retrained & hot-reloaded!"
                status_toast_until = time.time() + 4.0

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run_recognition()