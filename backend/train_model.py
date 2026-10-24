"""
SignSpeak ML Model Training Script
Prioritizes 100% real user-calibrated samples when available in collected_samples.json,
and falls back to SIGNS prototypes for uncalibrated letters.
"""
import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from backend.signs_data import SIGNS, ONE_HAND_LETTERS, TWO_HAND_LETTERS
except ImportError:
    from signs_data import SIGNS, ONE_HAND_LETTERS, TWO_HAND_LETTERS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COLLECTED_SAMPLES_FILE = os.path.join(DATA_DIR, "collected_samples.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def augment_features(base_features, num_samples=1000, rng=None, is_real=False, letter=""):
    if rng is None:
        rng = np.random.default_rng(42)

    base = np.array(base_features, dtype=np.float32)
    dim = len(base)
    augmented = np.zeros((num_samples, dim), dtype=np.float32)

    for i in range(num_samples):
        scale = rng.uniform(0.94, 1.06) if is_real else rng.uniform(0.91, 1.09)
        sample = base.copy()

        # Distances: features 0 to 8
        d_std = 0.028 if is_real else 0.038
        d_noise = rng.normal(0, d_std, size=9)
        sample[:9] = np.maximum(0.01, (sample[:9] * scale) + d_noise)

        # Angles: features 9 to 12
        a_std = 4.0 if is_real else 5.5
        a_noise = rng.normal(0, a_std, size=4)
        sample[9:13] = np.clip(sample[9:13] + a_noise, 0.0, 180.0)

        # Orientation Tilt: feature 13
        if dim > 13:
            t_std = 8.0 if is_real else (18.0 if letter == "U" else 8.5)
            t_noise = rng.normal(0, t_std)
            sample[13] = np.clip(sample[13] + t_noise, 0.0, 180.0)

        augmented[i] = sample

    return augmented


def _filter_steady_frames(vecs):
    """
    Drops the first few frames recorded right after pressing a keyboard key
    (when the user's hand was still moving into position) and removes statistical outliers.
    """
    if len(vecs) <= 10:
        return vecs
    # Take the most recent calibration batch (last 30 frames) and skip the first 6 transition frames
    recent = vecs[-30:] if len(vecs) >= 24 else vecs
    steady = recent[6:] if len(recent) > 12 else recent
    arr = np.array(steady, dtype=np.float32)
    median = np.median(arr, axis=0)
    dists = np.linalg.norm(arr[:, :9] - median[:9], axis=1)
    cutoff = np.percentile(dists, 85) + 1e-5
    filtered = [steady[i] for i in range(len(steady)) if dists[i] <= cutoff]
    return filtered if len(filtered) >= 5 else steady


def load_collected_samples():
    if os.path.exists(COLLECTED_SAMPLES_FILE):
        try:
            with open(COLLECTED_SAMPLES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def build_datasets(samples_per_class=1600):
    collected = load_collected_samples()
    rng = np.random.default_rng(42)

    X_one, y_one = [], []
    X_two, y_two = [], []

    # 1. ONE-HAND DATASET
    for letter in ONE_HAND_LETTERS:
        entry = SIGNS.get(letter, {})
        if "OneHand" not in entry:
            continue
        base = entry["OneHand"]
        alt_base = entry.get("AltOneHand")
        raw_user_vecs = [v for v in collected.get(letter, []) if len(v) == 14]
        user_vecs = _filter_steady_frames(raw_user_vecs)

        chunks = []
        if user_vecs:
            # Use 100% real calibrated samples for this letter so fake templates never override it!
            per_real = max(2, samples_per_class // len(user_vecs))
            for v in user_vecs:
                chunks.append(augment_features(v, num_samples=per_real, rng=rng, is_real=True, letter=letter))
            aug = np.vstack(chunks)[:samples_per_class]
            print(f"  [One-Hand '{letter}'] Trained on {len(user_vecs)} steady calibrated webcam frames.")
        elif alt_base is not None:
            half = samples_per_class // 2
            chunks.append(augment_features(base, num_samples=half, rng=rng, is_real=False, letter=letter))
            chunks.append(augment_features(alt_base, num_samples=samples_per_class - half, rng=rng, is_real=False, letter=letter))
            aug = np.vstack(chunks)
        else:
            aug = augment_features(base, num_samples=samples_per_class, rng=rng, is_real=False, letter=letter)

        X_one.append(aug)
        y_one.extend([letter] * len(aug))

    X_one = np.vstack(X_one)
    y_one = np.array(y_one)

    # 2. TWO-HAND DATASET WITH PERMUTATION INVARIANCE
    half_samples = samples_per_class // 2
    for letter in TWO_HAND_LETTERS:
        data = SIGNS.get(letter, {})
        if "Right" not in data or "Left" not in data:
            continue
        h_right = data["Right"]
        h_left = data["Left"]
        dy = data.get("dy", 0.1)

        raw_user_vecs = [v for v in collected.get(letter, []) if len(v) == 29]
        user_vecs = _filter_steady_frames(raw_user_vecs)

        if user_vecs:
            # Use 100% real calibrated samples for this two-hand letter (both hand permutations)
            combined_list = []
            per_real = max(4, samples_per_class // len(user_vecs))
            half_real = per_real // 2
            for vec in user_vecs:
                r_real, l_real, dy_real = vec[:14], vec[14:28], vec[28]
                ar = augment_features(r_real, num_samples=2 * half_real, rng=rng, is_real=True, letter=letter)
                al = augment_features(l_real, num_samples=2 * half_real, rng=rng, is_real=True, letter=letter)
                ady = np.maximum(0.0, dy_real + rng.normal(0, 0.045, size=(2 * half_real, 1)))
                p1 = np.hstack([ar[:half_real], al[:half_real], ady[:half_real]])
                p2 = np.hstack([al[half_real:], ar[half_real:], ady[half_real:]])
                combined_list.extend([p1, p2])
            combined = np.vstack(combined_list)[:samples_per_class]
            print(f"  [Two-Hand '{letter}'] Trained on {len(user_vecs)} steady calibrated webcam frames.")
        else:
            # Make uncalibrated D, P, K, Q distinct in synthetic templates
            if letter == "D":
                dy = 0.22
            elif letter == "P":
                dy = 0.42
            elif letter == "Q":
                dy = 0.78
            elif letter == "K":
                dy = 0.18

            aug_r = augment_features(h_right, num_samples=samples_per_class, rng=rng, is_real=False, letter=letter)
            aug_l = augment_features(h_left, num_samples=samples_per_class, rng=rng, is_real=False, letter=letter)
            dy_noise = np.maximum(0.0, dy + rng.normal(0, 0.07, size=(samples_per_class, 1)))

            perm1 = np.hstack([aug_r[:half_samples], aug_l[:half_samples], dy_noise[:half_samples]])
            perm2 = np.hstack([aug_l[half_samples:], aug_r[half_samples:], dy_noise[half_samples:]])
            combined = np.vstack([perm1, perm2])

        X_two.append(combined)
        y_two.extend([letter] * len(combined))

    X_two = np.vstack(X_two)
    y_two = np.array(y_two)

    return (X_two, y_two), (X_one, y_one)


def train_and_evaluate(name, X, y, filename):
    print("=" * 50)
    print(f"Training {name} Random Forest + ExtraTrees Classifier")
    print(f"Dataset: X={X.shape}, classes={len(np.unique(y))}")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=24,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=26,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    clf = VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",
        weights=[2, 1]
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=4))

    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(clf, model_path)
    print(f"Saved {name} model to {model_path}\n")
    return clf, acc


def train_all():
    print("Building datasets from Calibrated Webcam Samples + ISL Reference Chart...")
    (X_two, y_two), (X_one, y_one) = build_datasets(samples_per_class=1600)

    clf_two, acc_two = train_and_evaluate("Two-Hand", X_two, y_two, "model_two_hand.joblib")
    clf_one, acc_one = train_and_evaluate("One-Hand", X_one, y_one, "model_one_hand.joblib")

    collected = load_collected_samples()
    labels_metadata = {
        "two_hand_classes": sorted(list(np.unique(y_two))),
        "one_hand_classes": sorted(list(np.unique(y_one))),
        "calibrated_letters": sorted(list(collected.keys())),
        "features_per_hand": 14,
        "two_hand_feature_dim": 29,
        "two_hand_accuracy": float(acc_two),
        "one_hand_accuracy": float(acc_one),
        "total_classes": len(np.unique(y_two)) + len(np.unique(y_one))
    }
    labels_path = os.path.join(MODELS_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels_metadata, f, indent=4)

    print("Model training complete!")
    return acc_two, acc_one


if __name__ == "__main__":
    train_all()