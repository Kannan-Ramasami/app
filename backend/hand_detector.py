"""
Hand Detector abstraction supporting both MediaPipe Tasks (Python 3.13+)
and legacy MediaPipe Solutions (Python <= 3.12) with smooth VIDEO tracking.
"""
import os
import time
import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.is_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self._last_timestamp_ms = 0

        if self.is_legacy:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence
            )
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
            if not os.path.exists(task_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                urllib.request.urlretrieve(url, task_path)

            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = []

        if self.is_legacy:
            results = self.detector.process(rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, hand in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[i].classification[0].label
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
                    norm = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                    detections.append({
                        "label": label,
                        "points": points,
                        "landmarks_norm": norm,
                        "_raw_hand": hand
                    })
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.time() * 1000)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = now_ms

            results = self.detector.detect_for_video(mp_image, now_ms)
            if results.hand_landmarks and results.handedness:
                for i, hand in enumerate(results.hand_landmarks):
                    cat = results.handedness[i][0]
                    label = cat.category_name if hasattr(cat, "category_name") else cat.display_name
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                    norm = [(lm.x, lm.y, lm.z) for lm in hand]
                    detections.append({
                        "label": label,
                        "points": points,
                        "landmarks_norm": norm,
                        "_raw_hand": hand
                    })

        # Sort spatially left-to-right by wrist X so hand order doesn't randomly flip
        detections.sort(key=lambda d: d["points"][0][0])
        return detections

    def draw_landmarks(self, frame_bgr, detections):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for idx, det in enumerate(detections):
            pts = det["points"]
            line_color = (0, 255, 120) if idx == 0 else (0, 215, 255)
            for p1_idx, p2_idx in connections:
                cv2.line(frame_bgr, pts[p1_idx], pts[p2_idx], line_color, 2)
            for pt in pts:
                cv2.circle(frame_bgr, pt, 4, (0, 0, 255), -1)

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()