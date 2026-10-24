import cv2
import numpy as np

try:
    from backend.hand_detector import HandDetector
    from backend.signs_data import compute_13_features, compute_normalized_features, distance
except ImportError:
    from hand_detector import HandDetector
    from signs_data import compute_13_features, compute_normalized_features, distance


def main():
    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("=" * 40)
    print("SignSpeak - Hand Landmark Extractor")
    print("Outputs both Normalized & Raw feature vectors")
    print("Press q in camera window to exit.")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        detections = detector.process_frame(frame)
        detector.draw_landmarks(frame, detections)

        for det in detections:
            hand_label = det["label"]
            points = det["points"]
            raw = compute_13_features(points)
            norm = compute_normalized_features(points)
            palm = distance(points[0], points[9])

            print(f"\n--- {hand_label} HAND (Palm size: {palm:.1f}px) ---")
            print("Normalized (Scale-Invariant) 13 Features:")
            print("[" + ", ".join(f"{x:.3f}" if i < 9 else f"{x:.1f}" for i, x in enumerate(norm)) + "]")
            print("Raw Pixel Values:")
            print("[" + ", ".join(f"{x:.2f}" for x in raw) + "]")

        cv2.imshow("SignSpeak - Feature Extractor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
