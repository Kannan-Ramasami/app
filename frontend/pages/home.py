import os
import sys
import time
import json
import threading
from collections import deque, Counter
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Resolve import paths for backend and frontend packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(FRONTEND_DIR)

for path in [WORKSPACE_DIR, FRONTEND_DIR, os.path.join(WORKSPACE_DIR, "app-project")]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.icon_loader import load_icon

# Backend ML imports
try:
    from backend.hand_detector import HandDetector
    from backend.classifier import SignClassifier
    from backend.signs_data import (
        ONE_HAND_LETTERS,
        TWO_HAND_LETTERS,
        distance,
        compute_normalized_features,
    )
    from backend.train_model import train_all, COLLECTED_SAMPLES_FILE
except ImportError:
    from hand_detector import HandDetector
    from classifier import SignClassifier
    from signs_data import (
        ONE_HAND_LETTERS,
        TWO_HAND_LETTERS,
        distance,
        compute_normalized_features,
    )
    from train_model import train_all, COLLECTED_SAMPLES_FILE


def save_calibrated_batch(letter: str, batch_vecs: list[list[float]]) -> None:
    """Saves a steady 30-frame calibration batch for the specified letter."""
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


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller

        # --- ML Pipeline Components ---
        self.detector = None
        self.classifier = None
        self.cap = None
        self.is_running = False
        self._cam_tk_img = None
        self._after_id = None

        # Recognition tracking states
        self.pred_history: deque[str | None] = deque(maxlen=8)
        self.conf_history: deque[float] = deque(maxlen=8)
        self.last_two_hand_time = 0.0
        self.last_hand_seen_time = time.time()
        self.prev_time = time.time()
        self.fps = 0.0

        # --- Letter -> Word -> Sentence Formation States ---
        self.current_word = ""
        self.sentence_words: list[str] = []
        self.current_dwell_letter: str | None = None
        self.dwell_start_time = 0.0
        self.last_committed_letter: str | None = None
        self.last_commit_time = 0.0
        self.smoothed_letter: str | None = None
        self.smoothed_conf = 0.0

        # Calibration state
        self.calibrating_letter: str | None = None
        self.calib_countdown_until = 0.0
        self.calib_frames_left = 0
        self.calib_buffer: list[list[float]] = []
        self.status_toast = ""
        self.status_toast_until = 0.0
        self.is_retraining = False

        # --- Load Icons ---
        self.icon_settings = load_icon("settings", color="777777", size=24)
        self.icon_help = load_icon("help", color="777777", size=24)
        self.icon_arrow = load_icon("external-link", color="777777", size=18)
        self.icon_camera = load_icon("camera", color="242424", size=80)
        self.icon_globe = load_icon("globe", color="777777", size=32)
        self.icon_record = load_icon("record", color="777777", size=32)

        # --- Build UI Layout ---
        self._build_top_bar()
        self._build_main_content()
        self._build_bottom_bar()

        # Keyboard shortcuts
        self._bind_shortcuts()

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION
    # -------------------------------------------------------------------------
    def _build_top_bar(self):
        """Builds the sleek top navigation capsule."""
        self.top_bar_container = tk.Frame(self, bg="#121212")
        self.top_bar_container.pack(fill="x", padx=30, pady=(22, 10))

        self.top_bar = tk.Frame(
            self.top_bar_container,
            bg="#181818",
            highlightbackground="#242424",
            highlightthickness=1,
            bd=0,
        )
        self.top_bar.pack(fill="x", ipady=6)

        # Logo Frame
        self.logo_frame = tk.Frame(self.top_bar, bg="#181818")
        self.logo_frame.pack(side="left", padx=25, pady=8)

        self.logo_sign = tk.Label(
            self.logo_frame,
            text="SIGN",
            font=("Helvetica", 22, "bold"),
            fg="#E6E6E6",
            bg="#181818",
        )
        self.logo_sign.pack(side="left")

        self.logo_speak = tk.Label(
            self.logo_frame,
            text="SPEAK",
            font=("Helvetica", 22, "bold"),
            fg="#959595",
            bg="#181818",
        )
        self.logo_speak.pack(side="left")

        # Top Right Buttons Frame
        self.top_right_frame = tk.Frame(self.top_bar, bg="#181818")
        self.top_right_frame.pack(side="right", padx=20)

        # Contact button
        self.contact_btn = tk.Button(
            self.top_right_frame,
            text="Contact ",
            image=self.icon_arrow,
            compound="right",
            font=("Helvetica", 11, "bold"),
            fg="#959595",
            bg="#121212",
            activebackground="#1E1E1E",
            activeforeground="#E6E6E6",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            padx=16,
            pady=6,
            command=lambda: self.controller.show_frame("NotFoundPage"),
        )
        self.contact_btn.pack(side="right", padx=(10, 0))
        self._add_hover(self.contact_btn, "#1E1E1E", "#121212")

        # Help button
        self.help_btn = tk.Button(
            self.top_right_frame,
            image=self.icon_help,
            bg="#121212",
            activebackground="#1E1E1E",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            width=36,
            height=36,
            command=lambda: self.controller.show_frame("HelpPage"),
        )
        self.help_btn.pack(side="right", padx=5)
        self._add_hover(self.help_btn, "#1E1E1E", "#121212")

        # Settings button
        self.settings_btn = tk.Button(
            self.top_right_frame,
            image=self.icon_settings,
            bg="#121212",
            activebackground="#1E1E1E",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            width=36,
            height=36,
            command=lambda: self.controller.show_frame("SettingsPage"),
        )
        self.settings_btn.pack(side="right", padx=5)
        self._add_hover(self.settings_btn, "#1E1E1E", "#121212")

    def _build_main_content(self):
        """Builds the 2-column layout: Camera Feed (Left) & Translation Panels (Right)."""
        self.main_content = tk.Frame(self, bg="#121212")
        self.main_content.pack(fill="both", expand=True, padx=30, pady=10)

        self.main_content.grid_columnconfigure(0, weight=3)
        self.main_content.grid_columnconfigure(1, weight=2)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)

        # -------------------------------------------------------------
        # Left Panel (Camera)
        # -------------------------------------------------------------
        self.camera_panel = tk.Frame(
            self.main_content,
            bg="#181818",
            highlightbackground="#242424",
            highlightthickness=1,
            bd=0,
        )
        self.camera_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 15))

        # Placeholder widgets when camera is inactive
        self.camera_placeholder_frame = tk.Frame(self.camera_panel, bg="#181818")
        self.camera_placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.camera_icon_label = tk.Label(
            self.camera_placeholder_frame,
            image=self.icon_camera,
            bg="#181818",
        )
        self.camera_icon_label.pack(pady=(0, 12))

        self.camera_label = tk.Label(
            self.camera_placeholder_frame,
            text="Enable Camera To Use",
            font=("Helvetica", 16, "bold"),
            fg="#5E5E5E",
            bg="#181818",
        )
        self.camera_label.pack(pady=4)

        self.camera_sublabel = tk.Label(
            self.camera_placeholder_frame,
            text="Click START below to begin ISL translation",
            font=("Helvetica", 11),
            fg="#3E3E3E",
            bg="#181818",
        )
        self.camera_sublabel.pack(pady=2)

        # Active Video Display Label
        self.video_display_label = tk.Label(self.camera_panel, bg="#101010")

        # -------------------------------------------------------------
        # Right Top Panel (Letters / Words)
        # -------------------------------------------------------------
        self.letters_panel = tk.Frame(
            self.main_content,
            bg="#181818",
            highlightbackground="#242424",
            highlightthickness=1,
            bd=0,
        )
        self.letters_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 15))

        self._build_letters_panel_content()

        # -------------------------------------------------------------
        # Right Bottom Panel (Sentence)
        # -------------------------------------------------------------
        self.sentence_panel = tk.Frame(
            self.main_content,
            bg="#181818",
            highlightbackground="#242424",
            highlightthickness=1,
            bd=0,
        )
        self.sentence_panel.grid(row=1, column=1, sticky="nsew")

        self._build_sentence_panel_content()

    def _build_letters_panel_content(self):
        """Constructs live letter detection, dwell bar, and word formation UI."""
        header_frame = tk.Frame(self.letters_panel, bg="#181818")
        header_frame.pack(fill="x", padx=20, pady=(15, 6))

        title_label = tk.Label(
            header_frame,
            text="TRANSLATED LETTERS / WORDS",
            font=("Helvetica", 11, "bold"),
            fg="#7E7E7E",
            bg="#181818",
        )
        title_label.pack(side="left")

        # Active Sign Pill & Dwell Progress Row
        dwell_row = tk.Frame(self.letters_panel, bg="#181818")
        dwell_row.pack(fill="x", padx=20, pady=4)

        self.active_sign_pill = tk.Label(
            dwell_row,
            text="SIGN: --",
            font=("Helvetica", 13, "bold"),
            fg="#00E676",
            bg="#202020",
            padx=10,
            pady=3,
            highlightbackground="#2C2C2C",
            highlightthickness=1,
        )
        self.active_sign_pill.pack(side="left")

        self.active_conf_label = tk.Label(
            dwell_row,
            text="0%",
            font=("Helvetica", 11, "bold"),
            fg="#00D7FF",
            bg="#181818",
        )
        self.active_conf_label.pack(side="left", padx=10)

        # Dwell Progress Bar Canvas
        self.dwell_canvas = tk.Canvas(
            dwell_row,
            width=140,
            height=12,
            bg="#222222",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.dwell_canvas.pack(side="right")
        self.dwell_canvas_bar = self.dwell_canvas.create_rectangle(
            0, 0, 0, 12, fill="#00E676", width=0
        )

        # Current Word Box
        word_box = tk.Frame(
            self.letters_panel,
            bg="#141414",
            highlightbackground="#222222",
            highlightthickness=1,
        )
        word_box.pack(fill="both", expand=True, padx=20, pady=8)

        word_caption = tk.Label(
            word_box,
            text="ACTIVE WORD IN PROGRESS",
            font=("Helvetica", 9, "bold"),
            fg="#555555",
            bg="#141414",
        )
        word_caption.pack(anchor="w", padx=12, pady=(8, 2))

        self.current_word_label = tk.Label(
            word_box,
            text="_",
            font=("Helvetica", 26, "bold"),
            fg="#FFFFFF",
            bg="#141414",
            wraplength=380,
        )
        self.current_word_label.pack(anchor="w", padx=12, pady=(2, 8))

        # Compact Action Controls Row
        btn_row = tk.Frame(self.letters_panel, bg="#181818")
        btn_row.pack(fill="x", padx=20, pady=(4, 15))

        self.add_word_btn = tk.Button(
            btn_row,
            text="＋ Add Word [Space]",
            font=("Helvetica", 10, "bold"),
            fg="#E6E6E6",
            bg="#222222",
            activebackground="#2C2C2C",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#2E2E2E",
            cursor="hand2",
            padx=10,
            pady=6,
            command=self.commit_current_word,
        )
        self.add_word_btn.pack(side="left", padx=(0, 6))
        self._add_hover(self.add_word_btn, "#2C2C2C", "#222222")

        self.del_btn = tk.Button(
            btn_row,
            text="⌫ Delete [Bksp]",
            font=("Helvetica", 10, "bold"),
            fg="#D0D0D0",
            bg="#222222",
            activebackground="#2C2C2C",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#2E2E2E",
            cursor="hand2",
            padx=10,
            pady=6,
            command=self.delete_last_char_or_word,
        )
        self.del_btn.pack(side="left", padx=6)
        self._add_hover(self.del_btn, "#2C2C2C", "#222222")

        self.clear_btn = tk.Button(
            btn_row,
            text="✕ Clear",
            font=("Helvetica", 10, "bold"),
            fg="#FF7777",
            bg="#222222",
            activebackground="#331A1A",
            activeforeground="#FFAAAA",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#3E2222",
            cursor="hand2",
            padx=10,
            pady=6,
            command=self.clear_all,
        )
        self.clear_btn.pack(side="right")
        self._add_hover(self.clear_btn, "#331A1A", "#222222")

    def _build_sentence_panel_content(self):
        """Constructs translated natural sentence UI."""
        header_frame = tk.Frame(self.sentence_panel, bg="#181818")
        header_frame.pack(fill="x", padx=20, pady=(15, 6))

        title_label = tk.Label(
            header_frame,
            text="TRANSLATED SENTENCE",
            font=("Helvetica", 11, "bold"),
            fg="#7E7E7E",
            bg="#181818",
        )
        title_label.pack(side="left")

        self.copy_btn = tk.Button(
            header_frame,
            text="📋 Copy",
            font=("Helvetica", 9, "bold"),
            fg="#959595",
            bg="#181818",
            activebackground="#242424",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            padx=8,
            pady=2,
            command=self._copy_sentence_to_clipboard,
        )
        self.copy_btn.pack(side="right")
        self._add_hover(self.copy_btn, "#242424", "#181818")

        # Sentence Display Box
        sentence_box = tk.Frame(
            self.sentence_panel,
            bg="#141414",
            highlightbackground="#222222",
            highlightthickness=1,
        )
        sentence_box.pack(fill="both", expand=True, padx=20, pady=8)

        self.sentence_label = tk.Label(
            sentence_box,
            text="Point hands at camera to begin sign translation...",
            font=("Helvetica", 15),
            fg="#5E5E5E",
            bg="#141414",
            wraplength=440,
            justify="left",
            anchor="nw",
        )
        self.sentence_label.pack(fill="both", expand=True, padx=14, pady=12)

        # Bottom Tip & Status Bar
        status_footer = tk.Frame(self.sentence_panel, bg="#181818")
        status_footer.pack(fill="x", padx=20, pady=(2, 12))

        tip_label = tk.Label(
            status_footer,
            text="💡 Auto-space after 1.6s pause | Hold sign 1.0s to append letter",
            font=("Helvetica", 9),
            fg="#555555",
            bg="#181818",
        )
        tip_label.pack(side="left")

    def _build_bottom_bar(self):
        """Constructs the START/STOP toggle, Language, and Record calibration buttons."""
        self.bottom_bar = tk.Frame(self, bg="#121212")
        self.bottom_bar.pack(fill="x", padx=30, pady=(10, 25))

        # Start / Stop button (wide capsule)
        self.start_btn = tk.Button(
            self.bottom_bar,
            text="START",
            font=("Helvetica", 14, "bold"),
            fg="#00E676",
            bg="#181818",
            activebackground="#242424",
            activeforeground="#00FF88",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            height=2,
            command=self.toggle_start,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 15))
        self._add_hover(self.start_btn, "#222222", "#181818")

        # Language button (Circle-styled)
        self.lang_btn = tk.Button(
            self.bottom_bar,
            image=self.icon_globe,
            bg="#181818",
            activebackground="#242424",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            width=54,
            height=54,
            command=lambda: self.controller.show_frame("NotFoundPage"),
        )
        self.lang_btn.pack(side="left", padx=5)
        self._add_hover(self.lang_btn, "#242424", "#181818")

        # Record button (Circle-styled, opens calibration dialog)
        self.record_btn = tk.Button(
            self.bottom_bar,
            image=self.icon_record,
            bg="#181818",
            activebackground="#242424",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            width=54,
            height=54,
            command=self.open_calibrate_prompt,
        )
        self.record_btn.pack(side="left", padx=5)
        self._add_hover(self.record_btn, "#242424", "#181818")

    # -------------------------------------------------------------------------
    # KEYBOARD SHORTCUTS & HELPERS
    # -------------------------------------------------------------------------
    def _add_hover(self, widget, hover_bg, normal_bg):
        """Adds smooth background hover feedback to standard Tkinter buttons."""
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover_bg))
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal_bg))

    def _bind_shortcuts(self):
        """Binds global keys on the toplevel window."""
        top = self.winfo_toplevel()
        top.bind("<space>", self._handle_space_shortcut)
        top.bind("<BackSpace>", self._handle_backspace_shortcut)
        top.bind("<c>", self._handle_calibrate_shortcut)
        top.bind("<C>", self._handle_calibrate_shortcut)

    def _handle_space_shortcut(self, event):
        focused = self.focus_get()
        if isinstance(focused, tk.Entry):
            return
        self.commit_current_word()

    def _handle_backspace_shortcut(self, event):
        focused = self.focus_get()
        if isinstance(focused, tk.Entry):
            return
        self.delete_last_char_or_word()

    def _handle_calibrate_shortcut(self, event):
        focused = self.focus_get()
        if isinstance(focused, tk.Entry):
            return
        self.open_calibrate_prompt()

    # -------------------------------------------------------------------------
    # CAMERA & ML INTEGRATION LOOP
    # -------------------------------------------------------------------------
    def _init_models(self):
        """Initializes HandDetector and SignClassifier safely."""
        if self.classifier is None:
            try:
                self.classifier = SignClassifier()
            except Exception as e:
                print(f"Initial model not found ({e}). Auto-training...")
                train_all()
                self.classifier = SignClassifier()

        if self.detector is None:
            self.detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)

    def toggle_start(self):
        """Toggles real-time webcam feed and ML recognition loop."""
        self.is_running = not self.is_running
        if self.is_running:
            self._init_models()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.is_running = False
                messagebox.showerror(
                    "Camera Error",
                    "Could not access webcam. Please verify camera permissions.",
                )
                return

            self.start_btn.configure(
                text="STOP",
                fg="#FF5555",
                bg="#261616",
                highlightbackground="#552222",
            )
            self._add_hover(self.start_btn, "#331E1E", "#261616")

            # Switch placeholder with video display label
            self.camera_placeholder_frame.place_forget()
            self.video_display_label.pack(fill="both", expand=True)

            self.pred_history.clear()
            self.conf_history.clear()
            self.last_two_hand_time = 0.0
            self.last_hand_seen_time = time.time()
            self.prev_time = time.time()
            self.fps = 0.0

            # Launch non-blocking main thread loop
            self._process_camera_loop()
        else:
            self._stop_camera()

    def _stop_camera(self):
        """Stops the camera stream and restores UI to standby mode."""
        self.is_running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.start_btn.configure(
            text="START",
            fg="#00E676",
            bg="#181818",
            highlightbackground="#242424",
        )
        self._add_hover(self.start_btn, "#222222", "#181818")

        self.video_display_label.pack_forget()
        self.camera_placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.camera_label.configure(text="Enable Camera To Use", fg="#5E5E5E")

        self.active_sign_pill.configure(text="SIGN: --", fg="#5E5E5E")
        self.active_conf_label.configure(text="0%", fg="#5E5E5E")
        self.dwell_canvas.coords(self.dwell_canvas_bar, 0, 0, 0, 12)

    def _process_camera_loop(self):
        """Processes webcam frames, predicts ISL signs, and drives dwell formation."""
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._after_id = self.after(15, self._process_camera_loop)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        curr_time = time.time()

        # Update FPS
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-5, (curr_time - self.prev_time)))
        self.prev_time = curr_time

        # Run hand detector
        detections = self.detector.process_frame(frame)
        self.detector.draw_landmarks(frame, detections)
        num_hands = len(detections)

        raw_letter: str | None = None
        raw_conf = 0.0
        top_candidates: list[tuple[str, float]] = []
        current_two_vec: list[float] | None = None
        current_one_vec: list[float] | None = None

        # -------------------------------------------------------------
        # 1. Feature Computation & Classification
        # -------------------------------------------------------------
        if num_hands >= 2:
            self.last_two_hand_time = curr_time
            self.last_hand_seen_time = curr_time

            pts1 = detections[0]["points"]
            pts2 = detections[1]["points"]
            feat1 = compute_normalized_features(pts1)
            feat2 = compute_normalized_features(pts2)

            palm_ref = max(10.0, distance(pts1[0], pts1[9]), distance(pts2[0], pts2[9]))
            dy = abs(pts1[0][1] - pts2[0][1]) / palm_ref
            current_two_vec = list(feat1) + list(feat2) + [dy]

            raw_letter, raw_conf, top_candidates = self.classifier.predict(
                hand1=feat1,
                hand2=feat2,
                dy=dy,
                min_confidence=0.36,
                return_top_k=True,
                pts1=pts1,
                pts2=pts2,
            )
        elif num_hands == 1:
            self.last_hand_seen_time = curr_time
            pts = detections[0]["points"]
            feat = compute_normalized_features(pts)
            current_one_vec = list(feat)

            # 0.65s two-hand occlusion hold
            if curr_time - self.last_two_hand_time < 0.65:
                raw_letter = None
                raw_conf = 0.0
            else:
                raw_letter, raw_conf, top_candidates = self.classifier.predict(
                    hand1=feat,
                    hand2=None,
                    min_confidence=0.45,
                    return_top_k=True,
                    pts1=pts,
                )

        # -------------------------------------------------------------
        # 2. Calibration Sampling Handling
        # -------------------------------------------------------------
        if (
            self.calibrating_letter is not None
            and curr_time >= self.calib_countdown_until
            and self.calib_frames_left > 0
        ):
            req_two = self.calibrating_letter in TWO_HAND_LETTERS
            if req_two and current_two_vec is not None:
                self.calib_buffer.append(current_two_vec)
                self.calib_frames_left -= 1
            elif not req_two and current_one_vec is not None:
                self.calib_buffer.append(current_one_vec)
                self.calib_frames_left -= 1

            if self.calib_frames_left == 0:
                letter_done = self.calibrating_letter
                save_calibrated_batch(letter_done, self.calib_buffer)
                self.calibrating_letter = None
                self.calib_buffer = []
                self.status_toast = f"'{letter_done}' Saved! Retraining in background..."
                self.status_toast_until = curr_time + 4.5
                self._trigger_background_retrain(letter_done)

        # -------------------------------------------------------------
        # 3. 8-Frame Majority Smoothing Window
        # -------------------------------------------------------------
        if raw_letter is not None:
            self.pred_history.append(raw_letter)
            self.conf_history.append(raw_conf)
        else:
            self.pred_history.append(None)
            self.conf_history.append(0.0)

        active_preds = [p for p in self.pred_history if p is not None]
        smoothed_letter: str | None = None
        smoothed_conf = 0.0

        if len(active_preds) >= 4:
            counts = Counter(active_preds)
            most_common_letter, count = counts.most_common(1)[0]
            if count >= 4:
                smoothed_letter = most_common_letter
                letter_confs = [
                    c for p, c in zip(self.pred_history, self.conf_history) if p == most_common_letter
                ]
                smoothed_conf = float(np.mean(letter_confs)) if letter_confs else 0.0

        self.smoothed_letter = smoothed_letter
        self.smoothed_conf = smoothed_conf

        # -------------------------------------------------------------
        # 4. Letter Dwell & Word Formation Engine
        # -------------------------------------------------------------
        dwell_progress = 0.0
        if smoothed_letter is not None:
            if smoothed_letter != self.current_dwell_letter:
                # Switched to a new letter -> reset dwell timer immediately
                self.current_dwell_letter = smoothed_letter
                self.dwell_start_time = curr_time
                dwell_progress = 0.0
            else:
                # Steady hold on current letter
                hold_duration = curr_time - self.dwell_start_time
                is_duplicate = smoothed_letter == self.last_committed_letter
                time_since_commit = curr_time - self.last_commit_time

                if is_duplicate and time_since_commit < 1.2:
                    # Duplicate letter cooldown active
                    dwell_progress = 0.0
                else:
                    dwell_progress = min(1.0, max(0.0, hold_duration / 1.0))
                    if hold_duration >= 1.0:
                        # Dwell threshold met! Append letter to current word
                        self.current_word += smoothed_letter
                        self.last_committed_letter = smoothed_letter
                        self.last_commit_time = curr_time
                        self.dwell_start_time = curr_time
                        dwell_progress = 1.0
        else:
            self.current_dwell_letter = None
            self.dwell_start_time = 0.0
            dwell_progress = 0.0

        # -------------------------------------------------------------
        # 5. Word -> Sentence Auto-Space on Hand Pause
        # -------------------------------------------------------------
        if num_hands == 0 and self.current_word:
            if curr_time - self.last_hand_seen_time >= 1.6:
                # Automatically commit word to sentence after 1.6s pause
                self.sentence_words.append(self.current_word)
                self.current_word = ""
                self.last_committed_letter = None
                self.current_dwell_letter = None
                self.last_hand_seen_time = curr_time

        # -------------------------------------------------------------
        # 6. Render OpenCV HUD onto Frame
        # -------------------------------------------------------------
        self._draw_hud(
            frame,
            w,
            h,
            num_hands,
            smoothed_letter,
            smoothed_conf,
            dwell_progress,
            top_candidates,
            curr_time,
        )

        # -------------------------------------------------------------
        # 7. Push Frame to Tkinter & Update Panels
        # -------------------------------------------------------------
        panel_w = max(320, self.camera_panel.winfo_width())
        panel_h = max(240, self.camera_panel.winfo_height())
        scale = min(panel_w / w, panel_h / h)
        target_w = max(10, int(w * scale))
        target_h = max(10, int(h * scale))

        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_img)

        self.video_display_label.configure(image=photo)
        self._cam_tk_img = photo

        self._update_panels(dwell_progress)

        self._after_id = self.after(15, self._process_camera_loop)

    def _draw_hud(
        self,
        frame,
        w,
        h,
        num_hands,
        smoothed_letter,
        smoothed_conf,
        dwell_progress,
        top_candidates,
        curr_time,
    ):
        """Draws top status bar, detected sign badge, dwell bar, and live HUD elements."""
        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 44), (18, 18, 22), -1)
        cv2.putText(
            frame,
            "SignSpeak ISL | Dwell: 1.0s  Auto-Space: 1.6s",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            2,
        )

        # Hands Detected Badges
        h1_color = (0, 230, 100) if num_hands >= 1 else (70, 70, 75)
        h2_color = (0, 230, 100) if num_hands >= 2 else (70, 70, 75)
        cv2.putText(
            frame,
            "H1 [OK]" if num_hands >= 1 else "H1 [--]",
            (w - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            h1_color,
            2,
        )
        cv2.putText(
            frame,
            "H2 [OK]" if num_hands >= 2 else "H2 [--]",
            (w - 90, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            h2_color,
            2,
        )

        # Detected Letter Badge
        if smoothed_letter is not None:
            pct = int(smoothed_conf * 100)
            is_two = smoothed_letter in TWO_HAND_LETTERS and num_hands >= 2
            type_text = "Two Hands" if is_two else "One Hand"

            cv2.rectangle(frame, (25, 55), (265, 175), (18, 18, 22), -1)
            cv2.rectangle(frame, (25, 55), (265, 175), (0, 230, 100), 2)
            cv2.putText(
                frame,
                smoothed_letter,
                (45, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.6,
                (0, 255, 120),
                4,
            )
            cv2.putText(
                frame,
                f"{type_text} | {pct}%",
                (45, 164),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                1,
            )

            # Dwell Progress Bar
            cv2.rectangle(frame, (25, 182), (265, 196), (30, 30, 36), -1)
            bar_w = int(240 * dwell_progress)
            bar_color = (0, 255, 120) if dwell_progress >= 0.99 else (0, 220, 255)
            if bar_w > 0:
                cv2.rectangle(frame, (25, 182), (25 + bar_w, 196), bar_color, -1)
            cv2.rectangle(frame, (25, 182), (265, 196), (70, 70, 80), 1)

            status_str = f"Hold: {int(dwell_progress * 100)}%" if dwell_progress < 1.0 else "Committed!"
            cv2.putText(
                frame,
                status_str,
                (275, 193),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bar_color,
                1,
            )
        else:
            cv2.rectangle(frame, (25, 55), (265, 115), (24, 24, 28), -1)
            cv2.putText(
                frame,
                "Show ISL Sign...",
                (45, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (140, 140, 140),
                1,
            )

        # Top 3 Candidates
        if top_candidates:
            cand_str = "Top: " + "  |  ".join(
                [f"{l}: {int(c*100)}%" for l, c in top_candidates[:3]]
            )
            cv2.rectangle(frame, (20, h - 45), (460, h - 15), (18, 18, 22), -1)
            cv2.putText(
                frame,
                cand_str,
                (28, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 215, 255),
                1,
            )

        # Calibration & Toast Overlays
        if self.calibrating_letter is not None:
            if curr_time < self.calib_countdown_until:
                rem = self.calib_countdown_until - curr_time
                cv2.rectangle(frame, (25, 205), (550, 245), (0, 140, 255), -1)
                cv2.putText(
                    frame,
                    f"HOLD SIGN '{self.calibrating_letter}' STEADY... In {rem:.1f}s",
                    (35, 232),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
            else:
                req = 2 if self.calibrating_letter in TWO_HAND_LETTERS else 1
                cv2.rectangle(frame, (25, 205), (550, 245), (0, 180, 80), -1)
                cv2.putText(
                    frame,
                    f"RECORDING '{self.calibrating_letter}' ({req} Hand): {self.calib_frames_left} frames left",
                    (35, 232),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
        elif curr_time < self.status_toast_until:
            cv2.rectangle(frame, (25, 205), (550, 245), (32, 32, 42), -1)
            cv2.putText(
                frame,
                self.status_toast,
                (35, 232),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 180),
                2,
            )

        # FPS Display
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (w - 100, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )

    def _update_panels(self, dwell_progress: float):
        """Updates text labels, dwell canvas bar, and translated sentence display."""
        # Top-Right Panel: Active Sign & Dwell
        if self.smoothed_letter is not None:
            self.active_sign_pill.configure(
                text=f"SIGN: {self.smoothed_letter}",
                fg="#00E676",
            )
            self.active_conf_label.configure(
                text=f"{int(self.smoothed_conf * 100)}%",
                fg="#00D7FF",
            )
        else:
            self.active_sign_pill.configure(text="SIGN: --", fg="#5E5E5E")
            self.active_conf_label.configure(text="0%", fg="#5E5E5E")

        bar_pixels = int(140 * dwell_progress)
        self.dwell_canvas.coords(self.dwell_canvas_bar, 0, 0, bar_pixels, 12)
        bar_fill = "#00E676" if dwell_progress >= 0.99 else "#00D7FF"
        self.dwell_canvas.itemconfig(self.dwell_canvas_bar, fill=bar_fill)

        # Current Word Display
        if self.current_word:
            spaced = " ".join(self.current_word)
            self.current_word_label.configure(text=f"{spaced} _", fg="#FFFFFF")
        else:
            self.current_word_label.configure(text="_", fg="#5E5E5E")

        # Bottom-Right Panel: Translated Sentence
        formatted = self.get_formatted_sentence()
        if formatted:
            self.sentence_label.configure(text=formatted, fg="#E6E6E6", font=("Helvetica", 16, "bold"))
        else:
            self.sentence_label.configure(
                text="Point hands at camera to begin sign translation...",
                fg="#5E5E5E",
                font=("Helvetica", 14),
            )

    # -------------------------------------------------------------------------
    # WORD & SENTENCE MANIPULATION
    # -------------------------------------------------------------------------
    def get_formatted_sentence(self) -> str:
        """Formats words into a natural English sentence with current word at tail."""
        if not self.sentence_words and not self.current_word:
            return ""

        words = []
        for i, w in enumerate(self.sentence_words):
            if i == 0:
                words.append(w.capitalize())
            else:
                words.append(w.lower())

        sentence = " ".join(words)
        if self.current_word:
            tail = f"[{self.current_word}...]"
            if sentence:
                sentence = f"{sentence}  {tail}"
            else:
                sentence = tail
        return sentence

    def commit_current_word(self):
        """Appends active in-progress word to sentence."""
        if self.current_word:
            self.sentence_words.append(self.current_word)
            self.current_word = ""
            self.current_dwell_letter = None
            self.last_committed_letter = None
            self._update_panels(0.0)

    def delete_last_char_or_word(self):
        """Deletes last letter of active word, or last word of sentence."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence_words:
            self.sentence_words.pop()
        self.current_dwell_letter = None
        self.last_committed_letter = None
        self._update_panels(0.0)

    def clear_all(self):
        """Resets both word and sentence buffers."""
        self.current_word = ""
        self.sentence_words.clear()
        self.current_dwell_letter = None
        self.last_committed_letter = None
        self._update_panels(0.0)

    def _copy_sentence_to_clipboard(self):
        """Copies the constructed natural sentence to system clipboard."""
        raw_words = [w.capitalize() if i == 0 else w.lower() for i, w in enumerate(self.sentence_words)]
        text = " ".join(raw_words)
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_toast = "Sentence copied to clipboard!"
            self.status_toast_until = time.time() + 2.5

    # -------------------------------------------------------------------------
    # CALIBRATION & HOT-RELOAD DIALOG
    # -------------------------------------------------------------------------
    def open_calibrate_prompt(self):
        """Opens quick modal letter picker (A-Z) for recording and hot-reloading."""
        # Ensure camera is active
        if not self.is_running:
            self.toggle_start()

        dialog = tk.Toplevel(self)
        dialog.title("Calibrate ISL Letter (A–Z)")
        dialog.geometry("420x360")
        dialog.resizable(False, False)
        dialog.configure(bg="#181818")
        dialog.transient(self)
        dialog.grab_set()

        header = tk.Label(
            dialog,
            text="Calibrate ISL Sign (A–Z)",
            font=("Helvetica", 16, "bold"),
            fg="#FFFFFF",
            bg="#181818",
        )
        header.pack(pady=(16, 4))

        sub = tk.Label(
            dialog,
            text="Select a letter to record 30 frames with a 1.5s countdown:",
            font=("Helvetica", 10),
            fg="#959595",
            bg="#181818",
        )
        sub.pack(pady=(0, 12))

        # Letter Grid
        grid_frame = tk.Frame(dialog, bg="#181818")
        grid_frame.pack(padx=20, pady=5)

        letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        cols = 6
        for idx, char in enumerate(letters):
            r = idx // cols
            c = idx % cols
            btn = tk.Button(
                grid_frame,
                text=char,
                font=("Helvetica", 12, "bold"),
                fg="#00E676",
                bg="#222222",
                activebackground="#00E676",
                activeforeground="#121212",
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground="#2E2E2E",
                cursor="hand2",
                width=4,
                height=1,
                command=lambda ch=char: self._start_calibration_for_letter(ch, dialog),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)
            self._add_hover(btn, "#2C2C2C", "#222222")

        cancel_btn = tk.Button(
            dialog,
            text="Cancel",
            font=("Helvetica", 10),
            fg="#959595",
            bg="#181818",
            relief="flat",
            bd=0,
            cursor="hand2",
            command=dialog.destroy,
        )
        cancel_btn.pack(pady=(14, 0))

    def _start_calibration_for_letter(self, letter: str, dialog: tk.Toplevel):
        """Initiates 1.5s countdown and 30-frame recording for selected letter."""
        dialog.destroy()
        self.calibrating_letter = letter.upper()
        self.calib_countdown_until = time.time() + 1.5
        self.calib_frames_left = 30
        self.calib_buffer = []
        self.status_toast = f"Preparing to record '{letter}' in 1.5s..."
        self.status_toast_until = time.time() + 2.0

    def _trigger_background_retrain(self, letter: str):
        """Retrains ensemble model on background thread and hot-reloads classifier."""
        def _worker():
            try:
                train_all()
                if self.classifier is not None:
                    self.classifier.reload_models()
                self.status_toast = f"'{letter}' Calibrated & Model Reloaded!"
                self.status_toast_until = time.time() + 4.5
            except Exception as e:
                print(f"Error during retrain: {e}")
                self.status_toast = f"Retrain failed: {e}"
                self.status_toast_until = time.time() + 4.5

        threading.Thread(target=_worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # EXTERNAL API COMPATIBILITY HOOKS
    # -------------------------------------------------------------------------
    def update_camera_frame(self, frame_image):
        """Provided for external API compatibility."""
        if hasattr(self, "video_display_label"):
            self.video_display_label.configure(image=frame_image)
            self._cam_tk_img = frame_image

    def update_translated_text(self, letters_words, sentence):
        """Provided for external API compatibility."""
        self.current_word_label.configure(text=letters_words)
        self.sentence_label.configure(text=sentence)

    def cleanup(self):
        """Releases camera and detector resources on application exit."""
        self._stop_camera()
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
