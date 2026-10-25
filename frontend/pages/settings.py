import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(FRONTEND_DIR)
for path in [WORKSPACE_DIR, FRONTEND_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.ui_components import DarkPillButton, DarkRoundedCard

try:
    from backend.train_model import train_all
except ImportError:
    train_all = None


class SettingsPage(tk.Frame):
    """
    Settings & Pipeline Calibration Page.
    Features anti-aliased DarkRoundedCards, dark pill controls (zero white radiobuttons),
    and live parameter tuning for the ML detection & translation pipeline.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller

        self.status_toast = ""
        self.status_toast_until = 0.0

        # Build UI layout
        self._build_top_bar()
        self._build_content()
        self._build_footer()

    def _build_top_bar(self):
        top_container = tk.Frame(self, bg="#121212")
        top_container.pack(fill="x", padx=30, pady=(18, 8))

        top_bar = DarkRoundedCard(
            top_container,
            width=1100,
            height=62,
            radius=28,
            card_bg="#171717",
            border_color="#262626",
            pad=8,
        )
        top_bar.pack(fill="x")

        # Logo / Title inside capsule
        title_frame = tk.Frame(top_bar.body, bg="#171717")
        title_frame.pack(side="left", padx=20, pady=2)

        tk.Label(
            title_frame,
            text="SIGN",
            font=("Helvetica", 22, "bold"),
            fg="#FFFFFF",
            bg="#171717",
        ).pack(side="left")

        tk.Label(
            title_frame,
            text="SPEAK",
            font=("Helvetica", 22, "bold"),
            fg="#888888",
            bg="#171717",
        ).pack(side="left")

        tk.Label(
            title_frame,
            text="  /  Settings & Pipeline Calibration",
            font=("Helvetica", 14),
            fg="#666666",
            bg="#171717",
        ).pack(side="left")

        # Back to Home Button (Rounded Pill)
        back_btn = DarkPillButton(
            top_bar.body,
            text="← Back to Home",
            font=("Helvetica", 11, "bold"),
            fg="#E6E6E6",
            normal_bg="#141414",
            normal_border="#282828",
            hover_bg="#202020",
            hover_border="#383838",
            height=36,
            width=150,
            radius=18,
            command=lambda: self.controller.show_frame("HomePage"),
        )
        back_btn.pack(side="right", padx=16)

    def _build_content(self):
        container = tk.Frame(self, bg="#121212")
        container.pack(fill="both", expand=True, padx=30, pady=6)
        container.grid_columnconfigure(0, weight=1, minsize=535)
        container.grid_columnconfigure(1, weight=1, minsize=535)
        container.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # Left Card: Detection, Camera & Timing (Rounded Card)
        # -------------------------------------------------------------
        left_card = DarkRoundedCard(
            container,
            width=535,
            height=545,
            radius=24,
            card_bg="#171717",
            border_color="#262626",
            pad=20,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        tk.Label(
            left_card.body,
            text="Recognition & Camera Settings",
            font=("Helvetica", 15, "bold"),
            fg="#FFFFFF",
            bg="#171717",
        ).pack(anchor="w", pady=(0, 16))

        # Camera Index (Replaced native white radiobuttons with Dark Pill Buttons)
        cam_row = tk.Frame(left_card.body, bg="#171717")
        cam_row.pack(fill="x", pady=8)
        tk.Label(
            cam_row,
            text="Camera Device Index:",
            font=("Helvetica", 11, "bold"),
            fg="#CCCCCC",
            bg="#171717",
        ).pack(side="left")

        self.cam_var = tk.IntVar(value=self.controller.get_setting("camera_index", 0))
        self.cam_buttons = {}

        cam_btn_frame = tk.Frame(cam_row, bg="#171717")
        cam_btn_frame.pack(side="left", padx=12)

        for idx in (0, 1, 2):
            is_active = (self.cam_var.get() == idx)
            btn = DarkPillButton(
                cam_btn_frame,
                text=f"Cam {idx}",
                font=("Helvetica", 10, "bold"),
                fg="#00E676" if is_active else "#888888",
                normal_bg="#1E2A20" if is_active else "#202020",
                normal_border="#00E676" if is_active else "#2C2C2C",
                hover_bg="#26382A" if is_active else "#282828",
                hover_border="#00FF88" if is_active else "#383838",
                height=30,
                width=80,
                radius=15,
                command=lambda i=idx: self._select_camera(i),
            )
            btn.pack(side="left", padx=4)
            self.cam_buttons[idx] = btn

        # Min Detection Confidence Slider
        self.conf_slider, self.conf_val_label = self._create_slider_row(
            left_card.body,
            label_text="Detection Confidence Threshold:",
            from_=0.30,
            to=0.90,
            initial=self.controller.get_setting("min_detection_confidence", 0.50),
            fmt=lambda v: f"{int(float(v)*100)}%",
        )

        # Dwell Timing Slider
        self.dwell_slider, self.dwell_val_label = self._create_slider_row(
            left_card.body,
            label_text="Sign Dwell Hold Time:",
            from_=0.5,
            to=2.5,
            initial=self.controller.get_setting("dwell_time", 1.0),
            fmt=lambda v: f"{float(v):.1f} seconds",
        )

        # Auto-space Pause Slider
        self.pause_slider, self.pause_val_label = self._create_slider_row(
            left_card.body,
            label_text="Auto-Space Hand Pause Time:",
            from_=1.0,
            to=3.0,
            initial=self.controller.get_setting("autospace_pause", 1.6),
            fmt=lambda v: f"{float(v):.1f} seconds",
        )

        # Duplicate Letter Cooldown Slider
        self.cooldown_slider, self.cooldown_val_label = self._create_slider_row(
            left_card.body,
            label_text="Duplicate Letter Cooldown:",
            from_=0.8,
            to=2.5,
            initial=self.controller.get_setting("duplicate_cooldown", 1.2),
            fmt=lambda v: f"{float(v):.1f} seconds",
        )

        # -------------------------------------------------------------
        # Right Card: Audio Narration & ML Model Management (Rounded Card)
        # -------------------------------------------------------------
        right_card = DarkRoundedCard(
            container,
            width=535,
            height=545,
            radius=24,
            card_bg="#171717",
            border_color="#262626",
            pad=20,
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        tk.Label(
            right_card.body,
            text="Speech (TTS) & Model Management",
            font=("Helvetica", 15, "bold"),
            fg="#FFFFFF",
            bg="#171717",
        ).pack(anchor="w", pady=(0, 16))

        # Live Narration Toggle
        tts_row = tk.Frame(right_card.body, bg="#171717")
        tts_row.pack(fill="x", pady=8)
        tk.Label(
            tts_row,
            text="Live Audio Narration:",
            font=("Helvetica", 11, "bold"),
            fg="#CCCCCC",
            bg="#171717",
        ).pack(side="left")

        self.tts_var = tk.BooleanVar(value=self.controller.get_setting("tts_enabled", False))
        self.tts_toggle_btn = DarkPillButton(
            tts_row,
            text="ON (Active)" if self.tts_var.get() else "OFF (Muted)",
            font=("Helvetica", 10, "bold"),
            fg="#00E676" if self.tts_var.get() else "#888888",
            normal_bg="#1A261C" if self.tts_var.get() else "#202020",
            normal_border="#00E676" if self.tts_var.get() else "#2C2C2C",
            hover_bg="#26382A" if self.tts_var.get() else "#282828",
            hover_border="#00FF88" if self.tts_var.get() else "#383838",
            height=32,
            width=110,
            radius=16,
            command=self._toggle_tts_setting,
        )
        self.tts_toggle_btn.pack(side="right")

        # Speech Rate Slider
        self.rate_slider, self.rate_val_label = self._create_slider_row(
            right_card.body,
            label_text="Speech Rate (WPM):",
            from_=120,
            to=240,
            initial=self.controller.get_setting("tts_rate", 175),
            fmt=lambda v: f"{int(float(v))} WPM",
        )

        # Voice Test Button Row
        voice_test_row = tk.Frame(right_card.body, bg="#171717")
        voice_test_row.pack(fill="x", pady=(4, 14))

        test_btn = DarkPillButton(
            voice_test_row,
            text="🔊 Test Voice Audio",
            font=("Helvetica", 10, "bold"),
            fg="#00D7FF",
            normal_bg="#202020",
            normal_border="#2E2E2E",
            hover_bg="#282828",
            hover_border="#00FFFF",
            height=34,
            width=160,
            radius=17,
            command=lambda: self.controller.speak_text("SignSpeak speech synthesis is functioning normally."),
        )
        test_btn.pack(side="left")

        # Subtle Divider
        tk.Frame(right_card.body, bg="#242424", height=1).pack(fill="x", pady=14)

        # Model Ensemble Status
        tk.Label(
            right_card.body,
            text="ML Ensemble Models Status",
            font=("Helvetica", 12, "bold"),
            fg="#E6E6E6",
            bg="#171717",
        ).pack(anchor="w")

        tk.Label(
            right_card.body,
            text="• 26 ISL Classes Loaded (8 One-Hand, 19 Two-Hand)\n• Real-Time Spatial Normalization Active\n• Dynamic Calibration Hot-Reload Supported",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#171717",
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        # Retrain All Models Button
        retrain_btn = DarkPillButton(
            right_card.body,
            text="⚡ Retrain & Hot-Reload All Models",
            font=("Helvetica", 11, "bold"),
            fg="#00E676",
            normal_bg="#1E2A20",
            normal_border="#2E4A35",
            hover_bg="#283A2C",
            hover_border="#00FF88",
            height=38,
            width=260,
            radius=19,
            command=self._retrain_models,
        )
        retrain_btn.pack(anchor="w", pady=4)

    def _select_camera(self, idx: int):
        self.cam_var.set(idx)
        for i, btn in self.cam_buttons.items():
            is_sel = (i == idx)
            btn.set_text(f"Cam {i}", "#00E676" if is_sel else "#888888")
            btn.set_style(
                normal_bg="#1E2A20" if is_sel else "#202020",
                normal_border="#00E676" if is_sel else "#2C2C2C",
                hover_bg="#26382A" if is_sel else "#282828",
                hover_border="#00FF88" if is_sel else "#383838",
            )

    def _create_slider_row(self, parent, label_text, from_, to, initial, fmt):
        frame = tk.Frame(parent, bg="#171717")
        frame.pack(fill="x", pady=8)

        header = tk.Frame(frame, bg="#171717")
        header.pack(fill="x")

        tk.Label(
            header,
            text=label_text,
            font=("Helvetica", 10, "bold"),
            fg="#CCCCCC",
            bg="#171717",
        ).pack(side="left")

        val_label = tk.Label(
            header,
            text=fmt(initial),
            font=("Helvetica", 10, "bold"),
            fg="#00D7FF",
            bg="#171717",
        )
        val_label.pack(side="right")

        slider = tk.Scale(
            frame,
            from_=from_,
            to=to,
            resolution=0.05 if to <= 3.0 else 1.0,
            orient="horizontal",
            showvalue=0,
            bg="#171717",
            troughcolor="#111111",
            activebackground="#00E676",
            highlightthickness=0,
            bd=0,
            command=lambda v: val_label.configure(text=fmt(v)),
        )
        slider.set(initial)
        slider.pack(fill="x", pady=(4, 0))
        return slider, val_label

    def _toggle_tts_setting(self):
        new_val = not self.tts_var.get()
        self.tts_var.set(new_val)
        self.tts_toggle_btn.set_text("ON (Active)" if new_val else "OFF (Muted)", "#00E676" if new_val else "#888888")
        self.tts_toggle_btn.set_style(
            normal_bg="#1A261C" if new_val else "#202020",
            normal_border="#00E676" if new_val else "#2C2C2C",
        )

    def _build_footer(self):
        footer = tk.Frame(self, bg="#121212")
        footer.pack(fill="x", padx=30, pady=(6, 20))

        # Save Button (Rounded Pill)
        save_btn = DarkPillButton(
            footer,
            text="✓ Save & Apply Settings",
            font=("Helvetica", 12, "bold"),
            fg="#121212",
            normal_bg="#00E676",
            normal_border="#00FF88",
            hover_bg="#00FF88",
            hover_border="#33FF99",
            height=42,
            width=210,
            radius=21,
            command=self.save_settings,
        )
        save_btn.pack(side="right", padx=(10, 0))

        # Reset Defaults Button (Rounded Pill)
        reset_btn = DarkPillButton(
            footer,
            text="Restore Defaults",
            font=("Helvetica", 11),
            fg="#AAAAAA",
            normal_bg="#181818",
            normal_border="#282828",
            hover_bg="#242424",
            hover_border="#383838",
            height=42,
            width=150,
            radius=21,
            command=self.restore_defaults,
        )
        reset_btn.pack(side="right")

        # Feedback Toast Label
        self.toast_label = tk.Label(
            footer,
            text="",
            font=("Helvetica", 11, "bold"),
            fg="#00E676",
            bg="#121212",
        )
        self.toast_label.pack(side="left")

    def save_settings(self):
        self.controller.set_setting("camera_index", self.cam_var.get())
        self.controller.set_setting("min_detection_confidence", float(self.conf_slider.get()))
        self.controller.set_setting("dwell_time", float(self.dwell_slider.get()))
        self.controller.set_setting("autospace_pause", float(self.pause_slider.get()))
        self.controller.set_setting("duplicate_cooldown", float(self.cooldown_slider.get()))
        self.controller.set_setting("tts_enabled", self.tts_var.get())
        self.controller.set_setting("tts_rate", int(self.rate_slider.get()))

        self.toast_label.configure(text="✓ Settings saved and applied to live pipeline!", fg="#00E676")
        self.after(3000, lambda: self.toast_label.configure(text=""))

    def restore_defaults(self):
        self._select_camera(0)
        self.conf_slider.set(0.50)
        self.dwell_slider.set(1.0)
        self.pause_slider.set(1.6)
        self.cooldown_slider.set(1.2)
        self.tts_var.set(False)
        self.tts_toggle_btn.set_text("OFF (Muted)", "#888888")
        self.tts_toggle_btn.set_style(normal_bg="#202020", normal_border="#2C2C2C")
        self.rate_slider.set(175)
        self.save_settings()

    def _retrain_models(self):
        if train_all is None:
            messagebox.showerror("Error", "train_all function could not be imported.")
            return

        self.toast_label.configure(text="⚡ Training ensemble models in background...", fg="#00D7FF")

        def _worker():
            try:
                train_all()
                home = self.controller.pages.get("HomePage")
                if home and home.classifier:
                    home.classifier.reload_models()
                self.after(0, lambda: self.toast_label.configure(
                    text="✓ Retraining complete! Models hot-reloaded.", fg="#00E676"
                ))
            except Exception as e:
                self.after(0, lambda: self.toast_label.configure(
                    text=f"Retrain error: {e}", fg="#FF5555"
                ))
            self.after(4500, lambda: self.toast_label.configure(text=""))

        threading.Thread(target=_worker, daemon=True).start()
