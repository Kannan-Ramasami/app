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
from PIL import Image, ImageDraw, ImageTk

# Resolve import paths for backend and frontend packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(FRONTEND_DIR)
REAL_CURRENT = os.path.dirname(os.path.realpath(__file__))
REAL_FRONTEND = os.path.dirname(REAL_CURRENT)
REAL_WORKSPACE = os.path.dirname(REAL_FRONTEND)

for path in [
    WORKSPACE_DIR,
    FRONTEND_DIR,
    REAL_WORKSPACE,
    REAL_FRONTEND,
    os.path.join(WORKSPACE_DIR, "app-project"),
    os.path.join(REAL_WORKSPACE, "app-project"),
]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.icon_loader import load_icon
from utils.ui_components import DarkPillButton, DarkCircleButton, DarkRoundedCard, DarkPillFrame
from utils.translator import translate_text, LANG_CODES, LANG_VOICES

# Backend ML imports (strictly package imports for Pyrefly compatibility)
from backend.hand_detector import HandDetector
from backend.classifier import SignClassifier
from backend.signs_data import (
    SIGNS,
    ONE_HAND_LETTERS,
    TWO_HAND_LETTERS,
    distance,
    angle,
    compute_normalized_features,
    validate_one_hand,
)
from backend.train_model import train_all, COLLECTED_SAMPLES_FILE


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


PLACEHOLDERS: dict[str, str] = {
    "English": "Point hands at camera to begin sign translation...",
    "Hindi": "संकेत अनुवाद शुरू करने के लिए कैमरे की ओर हाथ करें...",
    "Tamil": "சைகை மொழிபெயர்ப்பைத் தொடங்க கேமராவை நோக்கி கைகளை வைக்கவும்...",
    "Telugu": "సంకేత అనువాదాన్ని ప్రారంభించడానికి కెమెరా వైపు చేతులు చూపించండి...",
    "Spanish": "Apunte las manos a la cámara para comenzar la traducción...",
    "French": "Pointez vos mains vers la caméra pour commencer la traduction...",
    "German": "Richten Sie die Hände auf die Kamera, um die Übersetzung zu starten...",
}


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

        # Translation Caching States
        self._last_sentence_text: str | None = None
        self._last_sentence_lang: str | None = None
        self._cached_sentence_translation: str = ""
        self._last_word_text: str | None = None
        self._last_word_lang: str | None = None
        self._cached_word_translation: str = ""

        # Calibration & Status state
        self.calibrating_letter: str | None = None
        self.calib_countdown_until = 0.0
        self.calib_frames_left = 0
        self.calib_buffer: list[list[float]] = []
        self.status_toast = ""
        self.status_toast_until = 0.0
        self.is_retraining = False

        # --- Load Icons (matching exact Figma mockup) ---
        self.icon_settings = load_icon("settings", color="777777", size=22)
        self.icon_help = load_icon("help", color="777777", size=22)
        self.icon_arrow = load_icon("external-link", color="777777", size=14)
        self.icon_camera = load_icon("camera", color="777777", size=80)
        self.icon_speaker_off = load_icon("speaker", color="777777", size=24)
        self.icon_speaker_on = load_icon("speaker", color="00E676", size=24)
        self.icon_globe = load_icon("globe", color="777777", size=26)
        self.icon_record = load_icon("record", color="777777", size=26)

        # --- Build UI Layout ---
        self._build_top_bar()
        self._build_main_content()
        self._build_bottom_bar()

        # Keyboard shortcuts
        self._bind_shortcuts()

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION
    # -------------------------------------------------------------------------
    def _add_hover(self, widget, hover_bg, normal_bg):
        """Adds smooth background hover feedback to standard Tkinter buttons."""
        widget.bind("<Enter>", lambda e: widget.configure(bg=hover_bg))
        widget.bind("<Leave>", lambda e: widget.configure(bg=normal_bg))

    def _build_top_bar(self):
        """Builds the sleek, anti-aliased top navigation rounded capsule."""
        self.top_bar_container = tk.Frame(self, bg="#121212")
        self.top_bar_container.pack(fill="x", padx=30, pady=(18, 8))

        self.top_bar = DarkRoundedCard(
            self.top_bar_container,
            width=1100,
            height=62,
            radius=28,
            card_bg="#171717",
            border_color="#262626",
            pad=8,
        )
        self.top_bar.pack(fill="x")

        # Logo Frame inside top bar body
        self.logo_frame = tk.Frame(self.top_bar.body, bg="#171717")
        self.logo_frame.pack(side="left", padx=18, pady=2)

        self.logo_sign = tk.Label(
            self.logo_frame,
            text="SIGN",
            font=("Helvetica", 22, "bold"),
            fg="#FFFFFF",
            bg="#171717",
        )
        self.logo_sign.pack(side="left")

        self.logo_speak = tk.Label(
            self.logo_frame,
            text="SPEAK",
            font=("Helvetica", 22, "bold"),
            fg="#888888",
            bg="#171717",
        )
        self.logo_speak.pack(side="left")

        # Top Right Buttons Frame
        self.top_right_frame = tk.Frame(self.top_bar.body, bg="#171717")
        self.top_right_frame.pack(side="right", padx=14)

        # Contact button (Pill capsule)
        self.contact_btn = DarkPillButton(
            self.top_right_frame,
            text="Contact ↗",
            font=("Helvetica", 11, "bold"),
            fg="#959595",
            normal_bg="#141414",
            normal_border="#282828",
            hover_bg="#202020",
            hover_border="#383838",
            height=34,
            width=110,
            radius=17,
            command=self.open_contact_dialog,
        )
        self.contact_btn.pack(side="right", padx=(10, 0))

        # Help button (Circular capsule)
        self.help_btn = DarkCircleButton(
            self.top_right_frame,
            icon_name="help",
            icon_color="777777",
            icon_size=20,
            size=36,
            normal_bg="#141414",
            normal_border="#282828",
            hover_bg="#202020",
            hover_border="#383838",
            command=lambda: self.controller.show_frame("HelpPage"),
        )
        self.help_btn.pack(side="right", padx=6)

        # Settings button (Circular capsule)
        self.settings_btn = DarkCircleButton(
            self.top_right_frame,
            icon_name="settings",
            icon_color="777777",
            icon_size=20,
            size=36,
            normal_bg="#141414",
            normal_border="#282828",
            hover_bg="#202020",
            hover_border="#383838",
            command=lambda: self.controller.show_frame("SettingsPage"),
        )
        self.settings_btn.pack(side="right", padx=6)

    def _build_main_content(self):
        """Builds the 2-column layout: Camera Card (Left) & Translation Cards (Right)."""
        self.main_content = tk.Frame(self, bg="#121212")
        self.main_content.pack(fill="both", expand=True, padx=30, pady=6)

        self.main_content.grid_columnconfigure(0, weight=11, minsize=560)
        self.main_content.grid_columnconfigure(1, weight=10, minsize=520)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)

        # -------------------------------------------------------------
        # Left Panel (Camera Card) - ALWAYS VISIBLE with rounded card
        # -------------------------------------------------------------
        self.camera_panel = DarkRoundedCard(
            self.main_content,
            width=560,
            height=580,
            radius=24,
            card_bg="#171717",
            border_color="#262626",
            pad=12,
        )
        self.camera_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 16))

        # Standby placeholder when camera is inactive
        self.camera_placeholder_frame = tk.Frame(self.camera_panel.body, bg="#171717")
        self.camera_placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.camera_icon_label = tk.Label(
            self.camera_placeholder_frame,
            image=self.icon_camera,
            bg="#171717",
        )
        self.camera_icon_label.pack(pady=(0, 16))

        self.camera_label = tk.Label(
            self.camera_placeholder_frame,
            text="Enable Camera To Use",
            font=("Helvetica", 16, "bold"),
            fg="#888888",
            bg="#171717",
        )
        self.camera_label.pack(pady=4)

        self.camera_sublabel = tk.Label(
            self.camera_placeholder_frame,
            text="Click START below to begin ISL translation",
            font=("Helvetica", 11),
            fg="#555555",
            bg="#171717",
        )
        self.camera_sublabel.pack(pady=2)

        # Active Video Display Label
        self.video_display_label = tk.Label(self.camera_panel.body, bg="#171717")

        # -------------------------------------------------------------
        # Right Top Panel (Translated Letters / Words)
        # -------------------------------------------------------------
        self.letters_panel = DarkRoundedCard(
            self.main_content,
            width=520,
            height=280,
            radius=24,
            card_bg="#171717",
            border_color="#262626",
            pad=14,
        )
        self.letters_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 12))

        self._build_letters_panel_content()

        # -------------------------------------------------------------
        # Right Bottom Panel (Translated Sentence)
        # -------------------------------------------------------------
        self.sentence_panel = DarkRoundedCard(
            self.main_content,
            width=520,
            height=288,
            radius=24,
            card_bg="#171717",
            border_color="#262626",
            pad=14,
        )
        self.sentence_panel.grid(row=1, column=1, sticky="nsew")

        self._build_sentence_panel_content()

    def _build_letters_panel_content(self):
        """Constructs live letter detection, dwell bar, word formation and the 3 action pills."""
        # Header Row
        header_frame = tk.Frame(self.letters_panel.body, bg="#171717")
        header_frame.pack(fill="x", padx=4, pady=(2, 6))

        title_label = tk.Label(
            header_frame,
            text="Translated Letters / Words",
            font=("Helvetica", 13, "bold"),
            fg="#E6E6E6",
            bg="#171717",
        )
        title_label.pack(side="left")

        # Top-Right Pill Capsule: Live Sign & Dwell Progress
        self.dwell_pill_frame = DarkPillFrame(
            header_frame,
            width=175,
            height=30,
            card_bg="#1E1E1E",
            border_color="#2A2A2A",
            pad_x=8,
            pad_y=3,
        )
        self.dwell_pill_frame.pack(side="right")

        self.active_sign_pill = tk.Label(
            self.dwell_pill_frame.body,
            text="SIGN: --",
            font=("Helvetica", 10, "bold"),
            fg="#777777",
            bg="#1E1E1E",
        )
        self.active_sign_pill.pack(side="left", padx=(0, 6))

        self.dwell_canvas = tk.Canvas(
            self.dwell_pill_frame.body,
            width=70,
            height=8,
            bg="#141414",
            highlightthickness=0,
            bd=0,
        )
        self.dwell_canvas.pack(side="left")
        self.dwell_canvas_bar = self.dwell_canvas.create_rectangle(
            0, 0, 0, 8, fill="#00E676", width=0
        )

        # Center Active Word Card (Rounded Inner Card)
        self.word_box = DarkRoundedCard(
            self.letters_panel.body,
            width=490,
            height=130,
            radius=16,
            card_bg="#131313",
            border_color="#202020",
            pad=6,
        )
        self.word_box.pack(fill="both", expand=True, padx=4, pady=6)

        self.current_word_label = tk.Label(
            self.word_box.body,
            text="_",
            font=("Helvetica", 28, "bold"),
            fg="#555555",
            bg="#131313",
            wraplength=460,
        )
        self.current_word_label.pack(expand=True, pady=(6, 2))

        self.current_word_trans_label = tk.Label(
            self.word_box.body,
            text="",
            font=("Helvetica", 12),
            fg="#00E676",
            bg="#131313",
        )
        self.current_word_trans_label.pack(pady=(0, 6))

        # Bottom Row: Exactly 3 Pill Buttons Matching the Mockup
        btn_row = tk.Frame(self.letters_panel.body, bg="#171717")
        btn_row.pack(fill="x", padx=4, pady=(6, 4))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        btn_row.grid_columnconfigure(2, weight=1)

        # Pill 1: Add Word
        self.add_word_btn = DarkPillButton(
            btn_row,
            text="＋ Add Word [Space]",
            font=("Helvetica", 10, "bold"),
            fg="#E6E6E6",
            normal_bg="#202020",
            normal_border="#282828",
            hover_bg="#2C2C2C",
            hover_border="#383838",
            height=36,
            radius=18,
            command=self.commit_current_word,
        )
        self.add_word_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        # Pill 2: Delete
        self.del_btn = DarkPillButton(
            btn_row,
            text="⌫ Delete [Bksp]",
            font=("Helvetica", 10, "bold"),
            fg="#D0D0D0",
            normal_bg="#202020",
            normal_border="#282828",
            hover_bg="#2C2C2C",
            hover_border="#383838",
            height=36,
            radius=18,
            command=self.delete_last_char_or_word,
        )
        self.del_btn.grid(row=0, column=1, sticky="ew", padx=6)

        # Pill 3: Clear
        self.clear_btn = DarkPillButton(
            btn_row,
            text="✕ Clear [Esc]",
            font=("Helvetica", 10, "bold"),
            fg="#FF7777",
            normal_bg="#202020",
            normal_border="#362222",
            hover_bg="#331A1A",
            hover_border="#552222",
            height=36,
            radius=18,
            command=self.clear_all,
        )
        self.clear_btn.grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def _build_sentence_panel_content(self):
        """Constructs translated natural sentence UI and top-right Copy pill."""
        # Header Row
        header_frame = tk.Frame(self.sentence_panel.body, bg="#171717")
        header_frame.pack(fill="x", padx=4, pady=(2, 6))

        title_label = tk.Label(
            header_frame,
            text="Translated Sentence",
            font=("Helvetica", 13, "bold"),
            fg="#E6E6E6",
            bg="#171717",
        )
        title_label.pack(side="left")

        self.lang_badge = tk.Label(
            header_frame,
            text="",
            font=("Helvetica", 10, "bold"),
            fg="#00E676",
            bg="#171717",
        )
        self.lang_badge.pack(side="left", padx=8)

        # Top-Right Pill: Copy Sentence
        self.copy_btn = DarkPillButton(
            header_frame,
            text="📋 Copy",
            font=("Helvetica", 10, "bold"),
            fg="#E6E6E6",
            normal_bg="#202020",
            normal_border="#282828",
            hover_bg="#2C2C2C",
            hover_border="#383838",
            height=28,
            width=85,
            radius=14,
            command=self._copy_sentence_to_clipboard,
        )
        self.copy_btn.pack(side="right")

        # Center Sentence Box (Rounded Inner Card)
        self.sentence_box = DarkRoundedCard(
            self.sentence_panel.body,
            width=490,
            height=140,
            radius=16,
            card_bg="#131313",
            border_color="#202020",
            pad=10,
        )
        self.sentence_box.pack(fill="both", expand=True, padx=4, pady=6)

        self.sentence_label = tk.Label(
            self.sentence_box.body,
            text="Point hands at camera to begin sign translation...",
            font=("Helvetica", 13),
            fg="#555555",
            bg="#131313",
            wraplength=460,
            justify="left",
            anchor="nw",
        )
        self.sentence_label.pack(fill="x", pady=(4, 2))

        self.sentence_sublabel = tk.Label(
            self.sentence_box.body,
            text="",
            font=("Helvetica", 10),
            fg="#777777",
            bg="#131313",
            anchor="w",
            wraplength=460,
            justify="left",
        )
        self.sentence_sublabel.pack(fill="x", pady=(2, 4))

        # Bottom Tip & Status Bar
        status_footer = tk.Frame(self.sentence_panel.body, bg="#171717")
        status_footer.pack(fill="x", padx=4, pady=(6, 2))

        self.tip_label = tk.Label(
            status_footer,
            text="💡 Auto-space after 1.6s pause | Hold sign 1.0s to append letter",
            font=("Helvetica", 9),
            fg="#555555",
            bg="#171717",
        )
        self.tip_label.pack(side="left")

        # Narration Indicator Pill
        self.narration_badge = tk.Label(
            status_footer,
            text="🔊 Narration: OFF",
            font=("Helvetica", 9, "bold"),
            fg="#777777",
            bg="#171717",
        )
        self.narration_badge.pack(side="right")

    def _build_bottom_bar(self):
        """Constructs wide START/STOP toggle and the 3 circular action buttons (Speaker, Globe, Record)."""
        self.bottom_bar = tk.Frame(self, bg="#121212")
        self.bottom_bar.pack(fill="x", padx=30, pady=(10, 25))

        # Start / Stop button (Wide capsule pill)
        self.start_btn = DarkPillButton(
            self.bottom_bar,
            text="START",
            font=("Helvetica", 14, "bold"),
            fg="#00E676",
            normal_bg="#181818",
            normal_border="#242424",
            hover_bg="#222222",
            hover_border="#333333",
            height=52,
            command=self.toggle_start,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 16))

        # Button 1: Speaker Icon (Live Audio Narration Toggle)
        self.speaker_btn = DarkCircleButton(
            self.bottom_bar,
            icon_name="speaker",
            icon_color="777777",
            icon_size=24,
            size=52,
            normal_bg="#181818",
            normal_border="#242424",
            hover_bg="#242424",
            hover_border="#383838",
            active_border="#00E676",
            active_bg="#1A261C",
            command=self.toggle_narration,
        )
        self.speaker_btn.pack(side="left", padx=5)

        # Button 2: Globe Icon (Language Selection Modal)
        self.lang_btn = DarkCircleButton(
            self.bottom_bar,
            icon_name="globe",
            icon_color="777777",
            icon_size=26,
            size=52,
            normal_bg="#181818",
            normal_border="#242424",
            hover_bg="#242424",
            hover_border="#383838",
            active_border="#00E676",
            active_bg="#1A261C",
            command=self.open_language_modal,
        )
        self.lang_btn.pack(side="left", padx=5)

        # Button 3: Record / Circle-Dot Icon (26-Letter Calibration Modal)
        self.record_btn = DarkCircleButton(
            self.bottom_bar,
            icon_name="record",
            icon_color="777777",
            icon_size=26,
            size=52,
            normal_bg="#181818",
            normal_border="#242424",
            hover_bg="#242424",
            hover_border="#383838",
            command=self.open_calibrate_prompt,
        )
        self.record_btn.pack(side="left", padx=5)

    # -------------------------------------------------------------------------
    # LIVE SPEECH NARRATION & SETTINGS SYNC
    # -------------------------------------------------------------------------
    def toggle_narration(self):
        """Toggles real-time speech narration and speaks current text if available."""
        current = self.controller.get_setting("tts_enabled", False)
        new_state = not current
        self.controller.set_setting("tts_enabled", new_state)
        self._update_speaker_ui(new_state)

        if new_state:
            text = self.get_formatted_sentence() or self.current_word
            if text:
                self.controller.speak_text(text)
            self.status_toast = "Live Voice Narration: ACTIVE"
        else:
            self.status_toast = "Live Voice Narration: MUTED"
        self.status_toast_until = time.time() + 2.5

    def _update_speaker_ui(self, is_active: bool):
        if hasattr(self, "speaker_btn"):
            self.speaker_btn.set_active(is_active)
            if is_active:
                self.narration_badge.configure(text="🔊 Narration: ON", fg="#00E676")
            else:
                self.narration_badge.configure(text="🔊 Narration: OFF", fg="#777777")

    def on_settings_updated(self):
        """Synchronizes settings when updated from SettingsPage."""
        tts_active = self.controller.get_setting("tts_enabled", False)
        self._update_speaker_ui(tts_active)
        self._last_word_text = None
        self._last_sentence_text = None
        self._update_panels(0.0)

    # -------------------------------------------------------------------------
    # KEYBOARD SHORTCUTS
    # -------------------------------------------------------------------------
    def _bind_shortcuts(self):
        """Binds global keys on the toplevel window."""
        top = self.winfo_toplevel()
        top.bind("<space>", self._handle_space_shortcut)
        top.bind("<BackSpace>", self._handle_backspace_shortcut)
        top.bind("<c>", self._handle_calibrate_shortcut)
        top.bind("<C>", self._handle_calibrate_shortcut)
        top.bind("<t>", self._handle_retrain_shortcut)
        top.bind("<T>", self._handle_retrain_shortcut)
        top.bind("<Escape>", self._handle_escape_shortcut)

    def _handle_space_shortcut(self, event):
        if not self.winfo_ismapped():
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.commit_current_word()

    def _handle_backspace_shortcut(self, event):
        if not self.winfo_ismapped():
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.delete_last_char_or_word()

    def _handle_calibrate_shortcut(self, event):
        if not self.winfo_ismapped():
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.open_calibrate_prompt()

    def _handle_retrain_shortcut(self, event):
        if not self.winfo_ismapped():
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self._trigger_background_retrain("Manual")

    def _handle_escape_shortcut(self, event):
        if not self.winfo_ismapped():
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)):
            return
        self.clear_all()

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
            min_det = self.controller.get_setting("min_detection_confidence", 0.5)
            self.detector = HandDetector(max_num_hands=2, min_detection_confidence=min_det)

    def toggle_start(self):
        """Toggles real-time webcam feed and ML recognition loop."""
        self.is_running = not self.is_running
        if self.is_running:
            self._init_models()
            cam_idx = self.controller.get_setting("camera_index", 0)
            self.cap = cv2.VideoCapture(cam_idx)
            if not self.cap.isOpened():
                self.is_running = False
                messagebox.showerror(
                    "Camera Error",
                    f"Could not access webcam device index {cam_idx}. Please verify camera permissions.",
                )
                return

            self.start_btn.set_text("STOP", new_fg="#FF5555")
            self.start_btn.set_style(
                normal_bg="#261616",
                normal_border="#552222",
                hover_bg="#331E1E",
                hover_border="#773333",
            )

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

        self.start_btn.set_text("START", new_fg="#00E676")
        self.start_btn.set_style(
            normal_bg="#181818",
            normal_border="#242424",
            hover_bg="#222222",
            hover_border="#333333",
        )

        self.video_display_label.pack_forget()
        self.camera_placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.camera_label.configure(text="Enable Camera To Use", fg="#888888")

        self.active_sign_pill.configure(text="SIGN: --", fg="#777777")
        self.dwell_canvas.coords(self.dwell_canvas_bar, 0, 0, 0, 8)

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
        dwell_target = self.controller.get_setting("dwell_time", 1.0)
        cooldown_target = self.controller.get_setting("duplicate_cooldown", 1.2)
        dwell_progress = 0.0

        if smoothed_letter is not None:
            if smoothed_letter != self.current_dwell_letter:
                # Switched to a new letter -> reset dwell timer immediately
                self.current_dwell_letter = smoothed_letter
                self.dwell_start_time = curr_time
                dwell_progress = 0.0
            else:
                # Steady hold on current letter
                is_duplicate = (smoothed_letter == self.last_committed_letter)
                time_since_commit = curr_time - self.last_commit_time

                if is_duplicate and time_since_commit < cooldown_target:
                    # Duplicate letter cooldown active
                    self.dwell_start_time = curr_time
                    dwell_progress = 0.0
                else:
                    hold_duration = curr_time - self.dwell_start_time
                    dwell_progress = min(1.0, max(0.0, hold_duration / dwell_target))
                    if hold_duration >= dwell_target:
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
        pause_target = self.controller.get_setting("autospace_pause", 1.6)
        if num_hands == 0 and self.current_word:
            if curr_time - self.last_hand_seen_time >= pause_target:
                # Automatically commit word to sentence after pause
                committed = self.current_word
                self.sentence_words.append(committed)
                self.current_word = ""
                self.last_committed_letter = None
                self.current_dwell_letter = None
                self.last_hand_seen_time = curr_time

                if self.controller.get_setting("tts_enabled", False):
                    active_lang = self.controller.get_setting("active_language", "English")
                    spoken = translate_text(committed, active_lang) if active_lang != "English" else committed
                    self.controller.speak_text(spoken, lang=active_lang)

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
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
        pil_img = Image.fromarray(frame_rgb)

        mask = Image.new("L", (target_w, target_h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=18, fill=255)
        pil_img.putalpha(mask)

        photo = ImageTk.PhotoImage(pil_img, master=self.winfo_toplevel())
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
        """Updates text labels, dwell canvas bar, word box and multi-language translated sentence."""
        # Top-Right Panel: Active Sign & Dwell Pill
        if self.smoothed_letter is not None:
            self.active_sign_pill.configure(
                text=f"SIGN: {self.smoothed_letter} ({int(self.smoothed_conf * 100)}%)",
                fg="#00E676",
            )
        else:
            self.active_sign_pill.configure(text="SIGN: --", fg="#777777")

        bar_pixels = int(70 * max(0.0, min(1.0, dwell_progress)))
        self.dwell_canvas.coords(self.dwell_canvas_bar, 0, 0, bar_pixels, 8)
        bar_fill = "#00E676" if dwell_progress >= 0.99 else "#00D7FF"
        self.dwell_canvas.itemconfig(self.dwell_canvas_bar, fill=bar_fill)

        active_lang = self.controller.get_setting("active_language", "English")

        # Update language header badge and bottom globe button active state
        if hasattr(self, "lang_badge"):
            if active_lang != "English":
                self.lang_badge.configure(text=f"•  {active_lang}", fg="#00E676")
                if hasattr(self, "lang_btn"):
                    self.lang_btn.set_active(True)
            else:
                self.lang_badge.configure(text="", fg="#777777")
                if hasattr(self, "lang_btn"):
                    self.lang_btn.set_active(False)

        # Current Word Display with Auto-Scaling Font to prevent any text overflow
        if self.current_word:
            spaced = " ".join(self.current_word)
            w_len = len(self.current_word)
            if w_len <= 8:
                word_font = ("Helvetica", 28, "bold")
            elif w_len <= 13:
                word_font = ("Helvetica", 22, "bold")
            elif w_len <= 18:
                word_font = ("Helvetica", 18, "bold")
            else:
                word_font = ("Helvetica", 15, "bold")

            self.current_word_label.configure(text=f"{spaced} _", fg="#FFFFFF", font=word_font)
            if active_lang != "English":
                if self.current_word != self._last_word_text or active_lang != self._last_word_lang:
                    self._last_word_text = self.current_word
                    self._last_word_lang = active_lang
                    self._cached_word_translation = translate_text(self.current_word, active_lang)
                self.current_word_trans_label.configure(text=f"→ {self._cached_word_translation} ({active_lang})")
            else:
                self.current_word_trans_label.configure(text="")
        else:
            self._last_word_text = None
            self._cached_word_translation = ""
            self.current_word_label.configure(text="_", fg="#555555", font=("Helvetica", 28, "bold"))
            self.current_word_trans_label.configure(text="")

        # Bottom-Right Panel: Multi-Language Translated Sentence
        full_sentence = " ".join(self.sentence_words)
        if self.current_word:
            full_sentence = f"{full_sentence} {self.current_word}".strip()

        if full_sentence:
            if active_lang != "English":
                if full_sentence != self._last_sentence_text or active_lang != self._last_sentence_lang:
                    self._last_sentence_text = full_sentence
                    self._last_sentence_lang = active_lang
                    self._cached_sentence_translation = translate_text(full_sentence, active_lang)

                translated = self._cached_sentence_translation or full_sentence
                s_len = len(translated)
                s_font = ("Helvetica", 17, "bold") if s_len < 55 else ("Helvetica", 14, "bold")
                self.sentence_label.configure(
                    text=translated,
                    fg="#FFFFFF",
                    font=s_font,
                )
                self.sentence_sublabel.configure(
                    text=f"Original (ISL): {full_sentence}",
                    fg="#777777",
                )
            else:
                s_len = len(full_sentence)
                s_font = ("Helvetica", 18, "bold") if s_len < 55 else ("Helvetica", 15, "bold")
                self.sentence_label.configure(
                    text=full_sentence,
                    fg="#FFFFFF",
                    font=s_font,
                )
                self.sentence_sublabel.configure(text="")
        else:
            self._last_sentence_text = None
            self._cached_sentence_translation = ""
            placeholder_text = PLACEHOLDERS.get(active_lang, PLACEHOLDERS["English"])
            self.sentence_label.configure(
                text=placeholder_text,
                fg="#555555",
                font=("Helvetica", 13),
            )
            if active_lang != "English":
                self.sentence_sublabel.configure(
                    text=f"Output: {active_lang}  •  Voice: {LANG_VOICES.get(active_lang, 'Default')}",
                    fg="#444444",
                )
            else:
                self.sentence_sublabel.configure(text="")

        # Update tip or toast notification
        if hasattr(self, "tip_label"):
            curr_time = time.time()
            if curr_time < self.status_toast_until:
                self.tip_label.configure(text=f"✨ {self.status_toast}", fg="#00E676")
            else:
                self.tip_label.configure(
                    text="💡 Auto-space after 1.6s pause | Hold sign 1.0s to append letter",
                    fg="#555555",
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
        """Appends active in-progress word to sentence with multi-lingual audio readout."""
        if self.current_word:
            committed = self.current_word
            self.sentence_words.append(committed)
            self.current_word = ""
            self.current_dwell_letter = None
            self.last_committed_letter = None
            self._update_panels(0.0)

            if self.controller.get_setting("tts_enabled", False):
                active_lang = self.controller.get_setting("active_language", "English")
                spoken = translate_text(committed, active_lang) if active_lang != "English" else committed
                self.controller.speak_text(spoken, lang=active_lang)

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
        """Copies translated sentence (in active language) to system clipboard."""
        full_sentence = " ".join(self.sentence_words)
        if self.current_word:
            full_sentence = f"{full_sentence} {self.current_word}".strip()

        active_lang = self.controller.get_setting("active_language", "English")
        if active_lang != "English":
            text_to_copy = translate_text(full_sentence, active_lang) or full_sentence
        else:
            text_to_copy = full_sentence

        if text_to_copy:
            self.clipboard_clear()
            self.clipboard_append(text_to_copy)
            self.status_toast = f"Copied ({active_lang}) to clipboard!"
            self.status_toast_until = time.time() + 2.5
            self.copy_btn.configure(text="✓ Copied!", fg="#00E676")
            self.after(2000, lambda: self.copy_btn.configure(text="📋 Copy", fg="#E6E6E6"))

    # -------------------------------------------------------------------------
    # MODAL DIALOGS (CONTACT, LANGUAGE, CALIBRATION)
    # -------------------------------------------------------------------------
    def open_contact_dialog(self):
        """Opens a sleek dark modal dialog with developer contact, feedback form, and repo info."""
        dialog = tk.Toplevel(self)
        dialog.title("Contact SignSpeak Team")
        dialog.geometry("460x420")
        dialog.resizable(False, False)
        dialog.configure(bg="#181818")
        dialog.transient(self)
        dialog.grab_set()

        header = tk.Label(
            dialog,
            text="Contact & Developer Support",
            font=("Helvetica", 16, "bold"),
            fg="#FFFFFF",
            bg="#181818",
        )
        header.pack(pady=(20, 6))

        tk.Label(
            dialog,
            text="SignSpeak AI • Real-Time ISL Translation Engine",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#181818",
        ).pack(pady=(0, 16))

        info_card = DarkRoundedCard(
            dialog,
            width=412,
            height=100,
            radius=16,
            card_bg="#141414",
            border_color="#242424",
            pad=10,
        )
        info_card.pack(fill="x", padx=24, pady=(0, 16))

        tk.Label(info_card.body, text="• GitHub:", font=("Helvetica", 10, "bold"), fg="#00D7FF", bg="#141414").grid(row=0, column=0, sticky="w", pady=2)
        tk.Label(info_card.body, text="github.com/Mmmaihdnsc/app-project", font=("Helvetica", 10), fg="#E6E6E6", bg="#141414").grid(row=0, column=1, sticky="w", padx=8, pady=2)

        tk.Label(info_card.body, text="• Support:", font=("Helvetica", 10, "bold"), fg="#00D7FF", bg="#141414").grid(row=1, column=0, sticky="w", pady=2)
        tk.Label(info_card.body, text="contact@signspeak.ai", font=("Helvetica", 10), fg="#E6E6E6", bg="#141414").grid(row=1, column=1, sticky="w", padx=8, pady=2)

        tk.Label(info_card.body, text="• Version:", font=("Helvetica", 10, "bold"), fg="#00D7FF", bg="#141414").grid(row=2, column=0, sticky="w", pady=2)
        tk.Label(info_card.body, text="v2.0 (Pure Tkinter + Ensemble ML)", font=("Helvetica", 10), fg="#E6E6E6", bg="#141414").grid(row=2, column=1, sticky="w", padx=8, pady=2)

        # Quick Feedback Message
        tk.Label(dialog, text="Send Quick Feedback:", font=("Helvetica", 11, "bold"), fg="#CCCCCC", bg="#181818").pack(anchor="w", padx=24, pady=(4, 4))
        fb_entry = tk.Entry(dialog, bg="#202020", fg="#FFFFFF", insertbackground="#00E676", font=("Helvetica", 11), relief="flat", highlightbackground="#282828", highlightthickness=1)
        fb_entry.pack(fill="x", padx=24, ipady=6, pady=(0, 16))
        fb_entry.insert(0, "Great ISL recognition accuracy!")

        status_msg = tk.Label(dialog, text="", font=("Helvetica", 10, "bold"), fg="#00E676", bg="#181818")
        status_msg.pack()

        btn_row = tk.Frame(dialog, bg="#181818")
        btn_row.pack(pady=10)

        def _send_feedback():
            status_msg.configure(text="✓ Thank you! Feedback recorded.")
            dialog.after(1500, dialog.destroy)

        send_btn = DarkPillButton(
            btn_row,
            text="Send Feedback",
            font=("Helvetica", 11, "bold"),
            fg="#121212",
            normal_bg="#00E676",
            normal_border="#00FF88",
            hover_bg="#00FF88",
            hover_border="#33FF99",
            height=36,
            width=140,
            radius=18,
            command=_send_feedback,
        )
        send_btn.pack(side="left", padx=8)

        close_btn = DarkPillButton(
            btn_row,
            text="Close",
            font=("Helvetica", 11),
            fg="#888888",
            normal_bg="#202020",
            normal_border="#282828",
            hover_bg="#2C2C2C",
            hover_border="#383838",
            height=36,
            width=90,
            radius=18,
            command=dialog.destroy,
        )
        close_btn.pack(side="left", padx=8)

    def open_language_modal(self):
        """Opens language selector modal with zero white controls and dark rounded selection pills."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Output Language")
        dialog.geometry("450x510")
        dialog.resizable(False, False)
        dialog.configure(bg="#181818")
        dialog.transient(self)
        dialog.grab_set()

        header = tk.Label(
            dialog,
            text="Select Output Language",
            font=("Helvetica", 16, "bold"),
            fg="#FFFFFF",
            bg="#181818",
        )
        header.pack(pady=(18, 4))

        tk.Label(
            dialog,
            text="Live translations and audio narration will use this language:",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#181818",
        ).pack(pady=(0, 14))

        languages = [
            ("English (Default)", "English"),
            ("Hindi (हिन्दी)", "Hindi"),
            ("Tamil (தமிழ்)", "Tamil"),
            ("Telugu (తెలుగు)", "Telugu"),
            ("Spanish (Español)", "Spanish"),
            ("French (Français)", "French"),
            ("German (Deutsch)", "German"),
        ]

        cur_active = self.controller.get_setting("active_language", "English")
        selected_lang = [cur_active]
        pill_btns: dict[str, DarkPillButton] = {}

        list_card = DarkRoundedCard(
            dialog,
            width=390,
            height=330,
            radius=18,
            card_bg="#141414",
            border_color="#242424",
            pad=8,
        )
        list_card.pack(padx=30, pady=(0, 12))

        def _on_lang_click(lang_key, label_txt):
            selected_lang[0] = lang_key
            for k, (btn, l_txt) in pill_btns.items():
                if k == lang_key:
                    btn.set_text(f"✓  {l_txt}", "#00E676")
                    btn.set_style(
                        normal_bg="#1A261C",
                        normal_border="#00E676",
                        hover_bg="#26382A",
                        hover_border="#00FF88",
                    )
                else:
                    btn.set_text(f"   {l_txt}", "#CCCCCC")
                    btn.set_style(
                        normal_bg="#161616",
                        normal_border="#242424",
                        hover_bg="#202020",
                        hover_border="#333333",
                    )

        for text_label, lang_val in languages:
            is_active = (lang_val == cur_active)
            btn_txt = f"✓  {text_label}" if is_active else f"   {text_label}"
            bg_col = "#1A261C" if is_active else "#161616"
            border_col = "#00E676" if is_active else "#242424"
            fg_col = "#00E676" if is_active else "#CCCCCC"

            btn = DarkPillButton(
                list_card.body,
                text=btn_txt,
                font=("Helvetica", 11, "bold" if is_active else "normal"),
                fg=fg_col,
                normal_bg=bg_col,
                normal_border=border_col,
                hover_bg="#26382A" if is_active else "#202020",
                hover_border="#00FF88" if is_active else "#333333",
                height=36,
                radius=18,
                command=lambda v=lang_val, l=text_label: _on_lang_click(v, l),
            )
            btn.pack(fill="x", pady=3)
            pill_btns[lang_val] = (btn, text_label)

        def _apply_lang():
            chosen = selected_lang[0]
            self.controller.set_setting("active_language", chosen)
            self.status_toast = f"Language set to {chosen}"
            self.status_toast_until = time.time() + 2.5
            self._last_word_text = None
            self._last_sentence_text = None
            self._update_panels(0.0)
            dialog.destroy()

        apply_btn = DarkPillButton(
            dialog,
            text="✓ Apply & Save",
            font=("Helvetica", 11, "bold"),
            fg="#121212",
            normal_bg="#00E676",
            normal_border="#00FF88",
            hover_bg="#00FF88",
            hover_border="#33FF99",
            height=38,
            width=180,
            radius=19,
            command=_apply_lang,
        )
        apply_btn.pack(pady=4)

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
            btn = DarkPillButton(
                grid_frame,
                text=char,
                font=("Helvetica", 12, "bold"),
                fg="#00E676",
                normal_bg="#222222",
                normal_border="#2E2E2E",
                hover_bg="#00E676",
                hover_border="#00FF88",
                radius=17,
                height=34,
                width=46,
                command=lambda ch=char: self._start_calibration_for_letter(ch, dialog),
            )
            btn.grid(row=r, column=c, padx=4, pady=4)

        cancel_btn = DarkPillButton(
            dialog,
            text="Cancel",
            font=("Helvetica", 10),
            fg="#959595",
            normal_bg="#181818",
            normal_border="#282828",
            hover_bg="#242424",
            hover_border="#383838",
            height=30,
            width=90,
            radius=15,
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
