"""
SignSpeak Landmark Coordinate Data & Geometry Functions
Defines benchmark coordinates for all 26 ISL letters strictly aligned with the ISL Reference Chart.
Supports 14 scale-invariant features per hand + inter-hand orientation and geometry.
"""

import math
import numpy as np

ONE_HAND_LETTERS = ["C", "I", "J", "L", "O", "U", "V", "W"]
TWO_HAND_LETTERS = [
    "A", "B", "D", "E", "F", "G", "H", "K", "M", "N",
    "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
]

FIST = [0.60, 0.15, 0.15, 0.15, 1.25, 0.75, 0.70, 0.65, 0.60, 30.0, 25.0, 25.0, 20.0, 20.0]
OPEN_PALM = [0.75, 0.20, 0.20, 0.20, 1.30, 2.00, 2.05, 1.95, 1.70, 175.0, 175.0, 175.0, 175.0, 10.0]
INDEX_UP = [0.80, 0.90, 0.12, 0.12, 0.90, 2.05, 0.80, 0.75, 0.70, 175.0, 35.0, 30.0, 25.0, 5.0]
INDEX_HORIZ = [0.75, 0.85, 0.12, 0.12, 0.85, 2.00, 0.75, 0.70, 0.65, 175.0, 30.0, 30.0, 25.0, 90.0]
INDEX_DIAG = [0.75, 0.85, 0.12, 0.12, 0.85, 2.00, 0.75, 0.70, 0.65, 175.0, 30.0, 30.0, 25.0, 45.0]
INDEX_VERT = [0.75, 0.85, 0.12, 0.12, 0.85, 2.00, 0.75, 0.70, 0.65, 175.0, 30.0, 30.0, 25.0, 5.0]
TWO_FINGERS = [0.75, 0.18, 1.10, 0.12, 0.85, 2.00, 2.05, 0.75, 0.70, 175.0, 175.0, 30.0, 25.0, 10.0]
THREE_FINGERS = [0.75, 0.20, 0.20, 0.95, 0.85, 2.00, 2.05, 1.95, 0.70, 175.0, 175.0, 170.0, 30.0, 10.0]
O_LOOP = [0.15, 0.08, 0.08, 0.12, 1.00, 0.98, 0.95, 0.90, 0.85, 105.0, 100.0, 100.0, 110.0, 20.0]
CURVED_HAND = [0.45, 0.15, 0.15, 0.15, 1.20, 1.35, 1.35, 1.30, 1.20, 115.0, 110.0, 110.0, 115.0, 25.0]
HOOK_INDEX = [0.65, 0.50, 0.12, 0.12, 0.85, 1.30, 0.75, 0.70, 0.65, 95.0, 35.0, 30.0, 25.0, 35.0]
PINKY_UP = [1.20, 0.12, 0.12, 0.85, 0.90, 0.75, 0.75, 0.75, 1.80, 30.0, 30.0, 30.0, 165.0, 10.0]
STEP_FINGERS = [0.60, 0.15, 0.15, 0.15, 1.00, 1.25, 1.25, 1.20, 1.15, 90.0, 90.0, 90.0, 90.0, 20.0]
V_SHAPE = [1.50, 0.85, 0.12, 0.12, 1.50, 2.00, 0.75, 0.70, 0.65, 175.0, 30.0, 30.0, 25.0, 15.0]
W_CLASP = [0.55, 0.20, 0.20, 0.20, 1.05, 1.20, 1.20, 1.15, 1.05, 100.0, 100.0, 100.0, 100.0, 65.0]

SIGNS = {
    # ----------------------------------------------------
    # ONE-HAND SIGNS
    # ----------------------------------------------------
    "C": {
        "OneHand": [0.75, 0.15, 0.15, 0.15, 1.25, 1.45, 1.45, 1.40, 1.30, 125.0, 120.0, 120.0, 125.0, 30.0]
    },
    "I": {
        "OneHand": [0.80, 0.90, 0.12, 0.12, 0.90, 2.05, 0.80, 0.75, 0.70, 175.0, 35.0, 35.0, 30.0, 5.0]
    },
    "J": {
        # Matches attached J reference image:
        # Template 1: Sideways/downward hand with index hooked horizontally at bottom (tilt ~95 deg)
        "OneHand": [0.75, 0.55, 0.14, 0.14, 0.95, 1.45, 0.78, 0.72, 0.68, 125.0, 45.0, 40.0, 35.0, 95.0],
        # Template 2: Sideways/downward fist with thumb/hook extended horizontally at bottom (tilt ~115 deg)
        "AltOneHand": [0.68, 0.22, 0.15, 0.15, 1.30, 0.92, 0.78, 0.72, 0.68, 70.0, 45.0, 40.0, 35.0, 115.0]
    },
    "L": {
        "OneHand": [1.80, 1.35, 0.15, 0.15, 1.60, 2.05, 0.75, 0.70, 0.65, 175.0, 30.0, 30.0, 25.0, 5.0]
    },
    "O": {
        "OneHand": [0.15, 0.08, 0.08, 0.12, 1.00, 0.98, 0.95, 0.90, 0.85, 90.0, 85.0, 85.0, 80.0, 25.0]
    },
    "U": {
        "OneHand": [1.50, 0.12, 0.12, 0.90, 1.50, 0.75, 0.75, 0.75, 1.85, 35.0, 35.0, 35.0, 175.0, 5.0],
        "AltOneHand": [1.15, 0.85, 0.15, 0.15, 1.35, 1.90, 0.75, 0.70, 0.65, 168.0, 35.0, 35.0, 30.0, 15.0]
    },
    "V": {
        "OneHand": [0.75, 0.85, 1.10, 0.12, 0.85, 2.00, 2.05, 0.75, 0.70, 175.0, 175.0, 30.0, 25.0, 5.0]
    },
    "W": {
        "OneHand": [0.75, 0.45, 0.45, 0.95, 0.85, 2.00, 2.05, 1.95, 0.70, 175.0, 175.0, 170.0, 30.0, 5.0],
        "Right": W_CLASP,
        "Left": W_CLASP,
        "dy": 0.15
    },

    # ----------------------------------------------------
    # TWO-HAND SIGNS (Untouched!)
    # ----------------------------------------------------
    "A": {"Right": FIST, "Left": FIST, "dy": 0.1},
    "B": {"Right": O_LOOP, "Left": O_LOOP, "dy": 0.1},
    "D": {"Right": CURVED_HAND, "Left": INDEX_UP, "dy": 0.1},
    "E": {"Right": INDEX_VERT, "Left": INDEX_UP, "dy": 0.1},
    "F": {"Right": TWO_FINGERS, "Left": TWO_FINGERS, "dy": 0.1},
    "G": {"Right": FIST, "Left": FIST, "dy": 1.8},
    "H": {"Right": OPEN_PALM, "Left": OPEN_PALM, "dy": 0.1},
    "K": {"Right": HOOK_INDEX, "Left": INDEX_UP, "dy": 0.1},
    "M": {"Right": THREE_FINGERS, "Left": OPEN_PALM, "dy": 0.2},
    "N": {"Right": TWO_FINGERS, "Left": OPEN_PALM, "dy": 0.2},
    "P": {"Right": O_LOOP, "Left": INDEX_UP, "dy": 0.1},
    "Q": {"Right": HOOK_INDEX, "Left": O_LOOP, "dy": 0.1},
    "R": {"Right": HOOK_INDEX, "Left": OPEN_PALM, "dy": 0.2},
    "S": {"Right": PINKY_UP, "Left": PINKY_UP, "dy": 0.1},
    "T": {"Right": INDEX_VERT, "Left": INDEX_HORIZ, "dy": 0.1},
    "X": {"Right": INDEX_DIAG, "Left": INDEX_DIAG, "dy": 0.1},
    "Y": {"Right": INDEX_VERT, "Left": V_SHAPE, "dy": 0.1},
    "Z": {"Right": STEP_FINGERS, "Left": OPEN_PALM, "dy": 0.2}
}


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_13_features(points):
    return [
        distance(points[4], points[8]),
        distance(points[8], points[12]),
        distance(points[12], points[16]),
        distance(points[16], points[20]),
        distance(points[4], points[0]),
        distance(points[8], points[0]),
        distance(points[12], points[0]),
        distance(points[16], points[0]),
        distance(points[20], points[0]),
        angle(points[5], points[6], points[8]),
        angle(points[9], points[10], points[12]),
        angle(points[13], points[14], points[16]),
        angle(points[17], points[18], points[20])
    ]


def compute_normalized_features(points):
    palm_size = distance(points[0], points[9])
    if palm_size < 10.0:
        palm_size = 10.0

    dists = [
        distance(points[4], points[8]) / palm_size,
        distance(points[8], points[12]) / palm_size,
        distance(points[12], points[16]) / palm_size,
        distance(points[16], points[20]) / palm_size,
        distance(points[4], points[0]) / palm_size,
        distance(points[8], points[0]) / palm_size,
        distance(points[12], points[0]) / palm_size,
        distance(points[16], points[0]) / palm_size,
        distance(points[20], points[0]) / palm_size
    ]

    angles = [
        angle(points[5], points[6], points[8]),
        angle(points[9], points[10], points[12]),
        angle(points[13], points[14], points[16]),
        angle(points[17], points[18], points[20])
    ]

    idx_dx = points[8][0] - points[5][0]
    idx_dy = points[8][1] - points[5][1]
    tilt = float(np.degrees(math.atan2(abs(idx_dx), -idx_dy)))

    return dists + angles + [tilt]


def validate_one_hand(letter, feat):
    if len(feat) < 13:
        return True

    dists = feat[:9]
    angles = feat[9:13]
    tilt = feat[13] if len(feat) > 13 else 20.0
    i_ang, m_ang, r_ang, p_ang = angles[0], angles[1], angles[2], angles[3]
    t_i_dist = dists[0]
    i_m_dist = dists[1]
    t_w_dist = dists[4]
    i_w_dist = dists[5]
    m_w_dist = dists[6]
    r_w_dist = dists[7]
    p_w_dist = dists[8]

    if letter == "C":
        if not (90.0 <= i_ang <= 155.0 and 85.0 <= m_ang <= 155.0 and
                85.0 <= r_ang <= 155.0 and 85.0 <= p_ang <= 155.0):
            return False
        if not (0.42 <= t_i_dist <= 1.32):
            return False
        return True

    elif letter == "U":
        mode_thumb_pinky = (
            p_ang >= 120.0 and p_w_dist >= 1.15
            and i_ang <= 100.0 and m_ang <= 100.0 and r_ang <= 100.0
        )
        mode_chart_u = (
            i_ang >= 130.0 and m_ang <= 95.0 and r_ang <= 95.0 and p_ang <= 105.0
            and 0.50 <= t_i_dist <= 1.40 and t_w_dist >= 1.05
        )
        mode_alt_u = (
            (i_ang >= 130.0 and p_ang >= 125.0 and m_ang <= 100.0 and r_ang <= 100.0)
            or (i_ang >= 135.0 and m_ang >= 135.0 and r_ang <= 95.0 and p_ang <= 95.0 and i_m_dist <= 0.42)
        )
        return bool(mode_thumb_pinky or mode_chart_u or mode_alt_u)

    elif letter == "I":
        if i_ang < 145.0 or i_w_dist < 1.5:
            return False
        if m_ang > 80.0 or r_ang > 80.0 or p_ang > 80.0:
            return False
        if t_i_dist > 1.3:
            return False
        return True

    elif letter == "W":
        if i_ang < 140.0 or m_ang < 140.0 or r_ang < 130.0:
            return False
        if p_ang > 90.0 or p_w_dist > 1.25:
            return False
        return True

    elif letter == "V":
        if i_ang < 140.0 or m_ang < 140.0:
            return False
        if r_ang > 90.0 or p_ang > 90.0:
            return False
        return True

    elif letter == "L":
        if i_ang < 140.0:
            return False
        if m_ang > 85.0 or r_ang > 85.0 or p_ang > 85.0:
            return False
        if t_w_dist < 1.15 or t_i_dist < 1.15:
            return False
        return True

    elif letter == "O":
        if t_i_dist > 0.42:
            return False
        if not (60.0 <= i_ang <= 130.0 and 60.0 <= m_ang <= 130.0 and 60.0 <= r_ang <= 130.0):
            return False
        return True

    elif letter == "J":
        # Matches attached J image:
        # Hand is tilted sideways/downward (tilt >= 42 deg), middle & ring are curled into palm,
        # and it is not a closed 'O' loop (t_i_dist > 0.38) or open 4-finger 'C'
        if tilt < 42.0:
            return False
        if m_w_dist > 1.45 or r_w_dist > 1.40:
            return False
        if m_ang > 125.0 or r_ang > 125.0:
            return False
        if t_i_dist < 0.36:
            return False
        return True

    return True