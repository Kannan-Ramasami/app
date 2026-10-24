"""
SignSpeak Real-Time ML Inference Classifier
Respects calibrated ML predictions, separates D / P / K from Q, and eliminates 1st/2nd place juggling.
"""
import os
import json
import math
import joblib
import numpy as np
from typing import Any, Optional, Sequence, Tuple, List, Union, overload, Literal

try:
    from backend.signs_data import validate_one_hand
except ImportError:
    from signs_data import validate_one_hand

TopKList = List[Tuple[str, float]]
PredictResult2 = Tuple[Optional[str], float]
PredictResult3 = Tuple[Optional[str], float, TopKList]
Point2D = Tuple[int, int]


def _pt_dist(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _seg_dist(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D) -> float:
    samples_a = [
        (int(a1[0] + t * (a2[0] - a1[0])), int(a1[1] + t * (a2[1] - a1[1])))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    samples_b = [
        (int(b1[0] + t * (b2[0] - b1[0])), int(b1[1] + t * (b2[1] - b1[1])))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    return min(_pt_dist(pa, pb) for pa in samples_a for pb in samples_b)


def _hand_scale(pts: Sequence[Point2D]) -> float:
    return max(
        12.0,
        _pt_dist(pts[0], pts[9]),
        _pt_dist(pts[0], pts[5]) * 0.95,
        _pt_dist(pts[5], pts[17]) * 1.25,
    )


def _analyze_hand(pts: Sequence[Point2D], feat: Sequence[float], scale: float) -> dict[str, Any]:
    t_i_dist = float(feat[0])
    t_w_dist = float(feat[4])
    i_w_dist = float(feat[5])
    m_w_dist = float(feat[6])
    r_w_dist = float(feat[7])
    p_w_dist = float(feat[8])
    i_ang, m_ang, r_ang, p_ang = float(feat[9]), float(feat[10]), float(feat[11]), float(feat[12])

    index_ext = (i_ang >= 125.0 and i_w_dist >= 1.20)
    middle_ext = (m_ang >= 125.0 and m_w_dist >= 1.20)
    ring_ext = (r_ang >= 120.0 and r_w_dist >= 1.15)
    pinky_ext = (p_ang >= 120.0 and p_w_dist >= 1.12)
    thumb_open = (t_w_dist >= 1.15 and t_i_dist >= 0.90)

    ext_count = int(index_ext) + int(middle_ext) + int(ring_ext) + int(pinky_ext)
    only_index = index_ext and not middle_ext and not ring_ext and not pinky_ext
    two_fingers = index_ext and middle_ext and not ring_ext and not pinky_ext
    three_fingers = index_ext and middle_ext and ring_ext and not pinky_ext
    open_palm = (ext_count >= 4) or (index_ext and middle_ext and ring_ext and p_ang >= 105.0)

    idx_dx = abs(pts[8][0] - pts[5][0])
    idx_dy = abs(pts[8][1] - pts[5][1])
    idx_angle_horiz = math.degrees(math.atan2(idx_dy, max(1e-5, idx_dx)))  # 0=horiz, 90=vert

    return {
        "t_i_dist": t_i_dist,
        "i_ang": i_ang,
        "m_ang": m_ang,
        "r_ang": r_ang,
        "p_ang": p_ang,
        "thumb_open": thumb_open,
        "index": index_ext,
        "middle": middle_ext,
        "ring": ring_ext,
        "pinky": pinky_ext,
        "ext_count": ext_count,
        "only_index": only_index,
        "two_fingers": two_fingers,
        "three_fingers": three_fingers,
        "open_palm": open_palm,
        "idx_angle_horiz": idx_angle_horiz,
        "idx_horiz": idx_angle_horiz < 46.0,
        "idx_vert": idx_angle_horiz >= 40.0,
        "idx_pointing_up": pts[8][1] < pts[5][1] - 0.08 * scale,
        "idx_hooked": (50.0 <= i_ang <= 152.0) and not middle_ext and not ring_ext and not pinky_ext,
        # True O-circle requires thumb tip and index tip to actually pinch close together (<= 0.42)
        "is_tight_o_loop": (t_i_dist <= 0.42) and (i_ang <= 135.0),
    }


def _add_boost(probs: np.ndarray, cls_idx: dict[str, int], letter: str, bonus: float) -> None:
    if letter in cls_idx:
        probs[cls_idx[letter]] += bonus


class SignClassifier:
    def __init__(self, models_dir: Optional[str] = None) -> None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.models_dir = models_dir
        self.two_hand_model_path = os.path.join(models_dir, "model_two_hand.joblib")
        self.one_hand_model_path = os.path.join(models_dir, "model_one_hand.joblib")
        self.labels_path = os.path.join(models_dir, "labels.json")
        self.clf_two: Any = None
        self.clf_one: Any = None
        self.metadata: dict[str, Any] = {}
        self.calibrated_letters: set[str] = set()
        self._ema_two: Optional[np.ndarray] = None
        self._ema_one: Optional[np.ndarray] = None
        self._locked_two_letter: Optional[str] = None
        self._locked_one_letter: Optional[str] = None
        self.reload_models()

    def reload_models(self) -> None:
        if not os.path.exists(self.two_hand_model_path) or not os.path.exists(self.one_hand_model_path):
            raise FileNotFoundError(
                f"Trained models not found in {self.models_dir}. Please run 'python backend/train_model.py' first."
            )
        self.clf_two = joblib.load(self.two_hand_model_path)
        self.clf_one = joblib.load(self.one_hand_model_path)
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.metadata = json.load(f)
            self.calibrated_letters = set(self.metadata.get("calibrated_letters", []))
        else:
            self.metadata = {}
            self.calibrated_letters = set()
        self._ema_two = None
        self._ema_one = None
        self._locked_two_letter = None
        self._locked_one_letter = None

    def _refine_two_hand_probs(
        self,
        probs: np.ndarray,
        classes: Sequence[str],
        pts1: Sequence[Point2D],
        pts2: Sequence[Point2D],
        feat1: Sequence[float],
        feat2: Sequence[float],
        dy: float,
    ) -> Optional[np.ndarray]:
        scale = max(_hand_scale(pts1), _hand_scale(pts2))

        min_cross_dist = min(_pt_dist(p1, p2) for p1 in pts1 for p2 in pts2) / scale
        palm_center_dist = _pt_dist(pts1[9], pts2[9]) / scale
        wrist_dy = abs(pts1[0][1] - pts2[0][1]) / scale

        # 1. BLOCK SEPARATE HANDS WHEN HELD FAR APART
        if min_cross_dist > 1.55 or palm_center_dist > 3.40:
            self._ema_two = None
            self._locked_two_letter = None
            return None

        adjusted = probs.copy().astype(np.float64)
        cls_idx = {str(c): i for i, c in enumerate(classes)}
        s1 = _analyze_hand(pts1, feat1, scale)
        s2 = _analyze_hand(pts2, feat2, scale)

        # 2. Prevent 'H' from triggering unless both hands are open palms
        if "H" in cls_idx:
            if not (s1["ext_count"] >= 3 and s2["ext_count"] >= 3 and min_cross_dist < 0.85):
                adjusted[cls_idx["H"]] *= 0.15

        # 3. CRITICAL FIX: Prevent 'Q' from stealing 'D', 'P', or 'K'!
        # In D, P, and K, one hand has a straight vertical index finger (i_ang >= 132) and wrists are at similar height.
        # True 'Q' has NO straight vertical index finger on the upper hand AND has a large vertical wrist offset (dy >= 0.48).
        either_has_straight_vert_index = (
            (s1["only_index"] and s1["idx_vert"] and s1["i_ang"] >= 132.0)
            or (s2["only_index"] and s2["idx_vert"] and s2["i_ang"] >= 132.0)
        )
        if "Q" in cls_idx and either_has_straight_vert_index and dy < 0.50:
            # Suppress false 'Q' when a straight vertical index stem (D, P, K, T, E) is shown!
            adjusted[cls_idx["Q"]] *= 0.12

        # 4. Gentle additive geometric boosts (adds +0.22 to +0.30 without wiping out calibrated ML probabilities)
        if s1["two_fingers"] and s2["two_fingers"] and min_cross_dist < 0.90:
            _add_boost(adjusted, cls_idx, "F", 0.25)

        for palm_s, other_s, palm_pts, other_pts in ((s1, s2, pts1, pts2), (s2, s1, pts2, pts1)):
            if palm_s["open_palm"]:
                tip8_to_palm = min(_pt_dist(other_pts[8], palm_pts[p]) for p in (0, 5, 6, 9, 10, 13, 17)) / scale
                if tip8_to_palm < 1.05:
                    if other_s["three_fingers"]:
                        _add_boost(adjusted, cls_idx, "M", 0.28)
                    elif other_s["two_fingers"]:
                        _add_boost(adjusted, cls_idx, "N", 0.28)
                    elif not other_s["middle"] and not other_s["ring"] and not other_s["pinky"] and (
                        other_s["idx_hooked"] or other_s["only_index"]
                    ):
                        _add_boost(adjusted, cls_idx, "R", 0.28)

        idx_seg_dist = _seg_dist(pts1[5], pts1[8], pts2[5], pts2[8]) / scale

        # Asymmetric index-stem signs: D, P, K, T, Q, E, Y
        for h_a, h_b, pts_a, pts_b in ((s1, s2, pts1, pts2), (s2, s1, pts2, pts1)):
            b_is_single_index = (
                (h_b["index"] or h_b["i_ang"] >= 118.0)
                and not h_b["middle"]
                and not h_b["ring"]
                and not h_b["pinky"]
            )
            if not b_is_single_index:
                continue

            # 'T': One horizontal index + one vertical index
            if (
                (h_a["index"] or h_a["i_ang"] >= 115.0)
                and not h_a["middle"]
                and not h_a["ring"]
                and h_a["idx_horiz"]
                and h_b["idx_vert"]
                and abs(h_b["idx_angle_horiz"] - h_a["idx_angle_horiz"]) >= 22.0
                and idx_seg_dist < 0.85
            ):
                _add_boost(adjusted, cls_idx, "T", 0.30)

            # 'D', 'P', 'K': Hand B is the straight vertical index stem (pts_b)
            if h_b["idx_vert"] and h_b["idx_pointing_up"] and h_b["i_ang"] >= 130.0:
                a_idx_to_b_top = min(_pt_dist(pts_a[8], pts_b[7]), _pt_dist(pts_a[8], pts_b[8])) / scale
                a_hook_to_b_mid = min(
                    _pt_dist(pts_a[6], pts_b[6]),
                    _pt_dist(pts_a[7], pts_b[6]),
                    _pt_dist(pts_a[8], pts_b[6]),
                    _pt_dist(pts_a[8], pts_b[5]),
                ) / scale
                a_thumb_to_b_stem = _seg_dist(pts_a[4], pts_a[3], pts_b[5], pts_b[8]) / scale
                tips_gap = _pt_dist(pts_a[8], pts_b[8]) / scale

                # 'K': Hand A has a bent/hooked single index touching the middle of Hand B's vertical index
                if (
                    not h_a["ring"]
                    and not h_a["pinky"]
                    and a_hook_to_b_mid < 0.90
                    and tips_gap > 0.24
                    and h_a["t_i_dist"] > 0.38
                ):
                    _add_boost(adjusted, cls_idx, "K", 0.32)

                # 'P' vs 'D': Hand A's index/loop touches near the upper stem of Hand B
                if a_idx_to_b_top < 0.95 or _seg_dist(pts_a[8], pts_a[6], pts_b[6], pts_b[8]) / scale < 0.75:
                    # In 'P', Hand A has a tighter thumb-index pinch (t_i_dist < 0.38) OR all fingers curled like O_LOOP
                    if h_a["t_i_dist"] < 0.38 or (h_a["m_ang"] >= 80.0 and h_a["t_i_dist"] < 0.48):
                        _add_boost(adjusted, cls_idx, "P", 0.32)
                    # In 'D', Hand A forms a wider C-arc (t_i_dist >= 0.38) with thumb touching lower stem
                    elif a_thumb_to_b_stem < 1.10 or h_a["t_i_dist"] >= 0.38:
                        _add_boost(adjusted, cls_idx, "D", 0.32)

            # 'Q': Hand A is an upper tight O-loop and Hand B is lower (`b_below_a > 0.35`) pointing up at it
            b_below_a = (pts_b[5][1] - pts_a[5][1]) / scale
            if b_below_a > 0.35 and h_a["is_tight_o_loop"] and dy >= 0.40:
                _add_boost(adjusted, cls_idx, "Q", 0.28)

        total = float(np.sum(adjusted))
        if total > 0:
            adjusted /= total

        # 5. EMA Temporal Smoothing across frames (stops 1st/2nd candidate juggling!)
        if self._ema_two is None or len(self._ema_two) != len(adjusted):
            self._ema_two = adjusted.copy()
        else:
            self._ema_two = 0.62 * self._ema_two + 0.38 * adjusted

        # 6. Hysteresis Lock: If a letter is already locked on screen, require a 10% lead to switch
        smoothed = self._ema_two.copy()
        if self._locked_two_letter in cls_idx:
            locked_i = cls_idx[self._locked_two_letter]
            best_i = int(np.argmax(smoothed))
            if best_i != locked_i and (smoothed[best_i] - smoothed[locked_i]) < 0.10:
                # Keep locked letter slightly ahead to prevent 1st/2nd place jitter
                smoothed[locked_i] = smoothed[best_i] + 0.02
                smoothed /= float(np.sum(smoothed))
            else:
                self._locked_two_letter = str(classes[best_i])
        else:
            best_i = int(np.argmax(smoothed))
            if smoothed[best_i] >= 0.38:
                self._locked_two_letter = str(classes[best_i])

        return smoothed

    def _refine_one_hand_probs(
        self,
        probs: np.ndarray,
        classes: Sequence[str],
        single_hand: Sequence[float],
        pts: Optional[Sequence[Point2D]] = None,
    ) -> np.ndarray:
        adjusted = probs.copy().astype(np.float64)
        cls_idx = {str(c): i for i, c in enumerate(classes)}

        # 1. 'J' detector matching attached J sign image:
        # All other 1-hand letters (C, I, L, O, U, V, W) are held upright (tilt < 42 deg).
        # In 'J', the hand is tilted sideways/downward with a bottom J-hook (index or thumb sticking out sideways)
        if "J" in cls_idx and len(single_hand) >= 14:
            t_i_dist = float(single_hand[0])
            t_w_dist = float(single_hand[4])
            i_w_dist = float(single_hand[5])
            m_w_dist = float(single_hand[6])
            r_w_dist = float(single_hand[7])
            m_ang = float(single_hand[10])
            r_ang = float(single_hand[11])
            p_ang = float(single_hand[12])
            tilt = float(single_hand[13])

            hand_angled_sideways_or_down = (tilt >= 48.0)
            if pts is not None:
                palm = max(12.0, _pt_dist(pts[0], pts[9]))
                # Wrist is higher than or level with the lower thumb/index hook
                wrist_above_hook = (pts[4][1] > pts[0][1] - 0.15 * palm) or (pts[8][1] > pts[0][1] - 0.15 * palm)
                hand_angled_sideways_or_down = hand_angled_sideways_or_down and wrist_above_hook

            middle_and_ring_curled = (m_w_dist < 1.38 and r_w_dist < 1.35 and m_ang < 118.0 and r_ang < 118.0)
            not_c_or_o = (t_i_dist > 0.40) and not (m_ang > 88.0 and r_ang > 88.0 and p_ang > 88.0)
            has_bottom_j_hook = (t_w_dist >= 0.85 or i_w_dist >= 0.88)

            if hand_angled_sideways_or_down and middle_and_ring_curled and not_c_or_o and has_bottom_j_hook:
                _add_boost(adjusted, cls_idx, "J", 0.48)

        # 2. Existing 'U' boost (untouched)
        if "U" in cls_idx and len(single_hand) >= 13 and validate_one_hand("U", single_hand):
            t_i_dist = float(single_hand[0])
            i_m_dist = float(single_hand[1])
            i_ang, m_ang, p_ang = float(single_hand[9]), float(single_hand[10]), float(single_hand[12])
            tilt = float(single_hand[13]) if len(single_hand) > 13 else 10.0
            is_clear_l = (i_ang >= 145.0 and m_ang < 85.0 and p_ang < 85.0 and t_i_dist >= 1.38)
            is_clear_v = (i_ang >= 145.0 and m_ang >= 145.0 and i_m_dist >= 0.44)
            if not is_clear_l and not is_clear_v and tilt < 45.0:
                _add_boost(adjusted, cls_idx, "U", 0.30)

        total = float(np.sum(adjusted))
        if total > 0:
            adjusted /= total

        if self._ema_one is None or len(self._ema_one) != len(adjusted):
            self._ema_one = adjusted.copy()
        else:
            self._ema_one = 0.65 * self._ema_one + 0.35 * adjusted

        return self._ema_one.copy()

    @overload
    def predict(
        self,
        hand1: Optional[Sequence[float]] = None,
        hand2: Optional[Sequence[float]] = None,
        dy: float = 0.1,
        min_confidence: float = 0.38,
        *,
        return_top_k: Literal[True],
        pts1: Optional[Sequence[Point2D]] = None,
        pts2: Optional[Sequence[Point2D]] = None,
        **kwargs: Any,
    ) -> PredictResult3: ...

    @overload
    def predict(
        self,
        hand1: Optional[Sequence[float]] = None,
        hand2: Optional[Sequence[float]] = None,
        dy: float = 0.1,
        min_confidence: float = 0.38,
        return_top_k: Literal[False] = False,
        pts1: Optional[Sequence[Point2D]] = None,
        pts2: Optional[Sequence[Point2D]] = None,
        **kwargs: Any,
    ) -> PredictResult2: ...

    def predict(
        self,
        hand1: Optional[Sequence[float]] = None,
        hand2: Optional[Sequence[float]] = None,
        dy: float = 0.1,
        min_confidence: float = 0.38,
        return_top_k: bool = False,
        pts1: Optional[Sequence[Point2D]] = None,
        pts2: Optional[Sequence[Point2D]] = None,
        **kwargs: Any,
    ) -> Union[PredictResult2, PredictResult3]:
        top_candidates: TopKList = []

        if hand1 is not None and hand2 is not None:
            vec1 = np.hstack([hand1, hand2, [dy]]).reshape(1, -1)
            vec2 = np.hstack([hand2, hand1, [dy]]).reshape(1, -1)

            probs1 = self.clf_two.predict_proba(vec1)[0]
            probs2 = self.clf_two.predict_proba(vec2)[0]
            fused_probs = np.maximum(probs1, probs2)

            if pts1 is not None and pts2 is not None:
                refined = self._refine_two_hand_probs(
                    fused_probs, self.clf_two.classes_, pts1, pts2, hand1, hand2, dy
                )
                if refined is None:
                    if return_top_k:
                        return None, 0.0, []
                    return None, 0.0
                fused_probs = refined

            sorted_idx = np.argsort(fused_probs)[::-1]
            for i in sorted_idx[:3]:
                top_candidates.append((str(self.clf_two.classes_[int(i)]), float(fused_probs[int(i)])))

            best_two_letter, best_two_conf = top_candidates[0]

            if best_two_conf >= min_confidence:
                if return_top_k:
                    return best_two_letter, best_two_conf, top_candidates
                return best_two_letter, best_two_conf

            if return_top_k:
                return None, best_two_conf, top_candidates
            return None, best_two_conf

        single_hand = hand1 if hand1 is not None else hand2
        if single_hand is not None:
            feat_14 = np.array(single_hand).reshape(1, -1)
            probs = self.clf_one.predict_proba(feat_14)[0]

            probs = self._refine_one_hand_probs(
                probs, self.clf_one.classes_, single_hand
            )

            sorted_indices = np.argsort(probs)[::-1]
            for i in sorted_indices[:3]:
                top_candidates.append((str(self.clf_one.classes_[int(i)]), float(probs[int(i)])))

            top_letter, top_conf = top_candidates[0]
            second_conf = top_candidates[1][1] if len(top_candidates) > 1 else 0.0
            margin = top_conf - second_conf

            if top_conf >= 0.45 and margin >= 0.08:
                if validate_one_hand(top_letter, single_hand) or top_letter in self.calibrated_letters:
                    if return_top_k:
                        return top_letter, top_conf, top_candidates
                    return top_letter, top_conf

            if return_top_k:
                return None, top_conf, top_candidates
            return None, top_conf

        if return_top_k:
            return None, 0.0, []
        return None, 0.0