import os
import sys
import tkinter as tk

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(FRONTEND_DIR)
for path in [WORKSPACE_DIR, FRONTEND_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.ui_components import DarkPillButton, DarkRoundedCard


class HelpPage(tk.Frame):
    """
    Help & ISL Reference Page with completely rounded cards,
    invisible scrollbar (smooth trackpad & keyboard navigation),
    and pure dark aesthetics.
    """
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller

        self._build_top_bar()
        self._build_content()

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

        # Logo / Title inside rounded capsule
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
            text="  /  Help & Indian Sign Language Guide",
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
        # Create an invisible scrollable canvas without any visible scrollbar widget
        canvas = tk.Canvas(self, bg="#121212", highlightthickness=0, bd=0)
        scrollable_frame = tk.Frame(canvas, bg="#121212")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1080)
        canvas.pack(fill="both", expand=True, padx=30, pady=(4, 20))

        # Smooth trackpad & discrete mousewheel scrolling (no visible scrollbar)
        def _on_mousewheel(event):
            if not self.winfo_ismapped():
                return
            delta = event.delta
            if abs(delta) >= 120:
                steps = int(-1 * (delta / 120) * 3)
            else:
                steps = int(-1 * delta)
            if steps != 0:
                canvas.yview_scroll(steps, "units")
            return "break"

        self.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units") if self.winfo_ismapped() else None)
        self.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units") if self.winfo_ismapped() else None)

        # Keyboard arrow and page scrolling
        def _on_key(event):
            if not self.winfo_ismapped():
                return
            k = event.keysym
            if k == "Up":
                canvas.yview_scroll(-2, "units")
            elif k == "Down":
                canvas.yview_scroll(2, "units")
            elif k in ("Prior", "Page_Up"):
                canvas.yview_scroll(-1, "pages")
            elif k in ("Next", "Page_Down"):
                canvas.yview_scroll(1, "pages")
            elif k == "Home":
                canvas.yview_moveto(0.0)
            elif k == "End":
                canvas.yview_moveto(1.0)

        self.bind_all("<KeyPress-Up>", _on_key)
        self.bind_all("<KeyPress-Down>", _on_key)
        self.bind_all("<KeyPress-Prior>", _on_key)
        self.bind_all("<KeyPress-Next>", _on_key)
        self.bind_all("<KeyPress-Home>", _on_key)
        self.bind_all("<KeyPress-End>", _on_key)

        # -------------------------------------------------------------
        # Section 1: Quick-Start Guide (Rounded Card)
        # -------------------------------------------------------------
        card1 = self._create_card(scrollable_frame, "Quick-Start Workflow")
        steps = [
            ("1. Start Stream", "Click the wide START button at the bottom to initialize webcam hand tracking."),
            ("2. Hold Sign Steady", "Show your hand posture to the camera. Look at the HUD box for the live detected sign."),
            ("3. Dwell 1.0s to Form Words", "Hold the sign steadily for 1.0 second. The dwell bar fills up and commits the letter."),
            ("4. Auto-Space After 1.6s Pause", "Drop or lower your hands for 1.6s. The word automatically commits into the sentence with a space."),
            ("5. Speech Narration", "Click the Speaker icon (🔊) in the bottom bar to toggle real-time voice readout in chosen language."),
        ]
        for title, desc in steps:
            row = tk.Frame(card1, bg="#171717")
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"• {title}:", font=("Helvetica", 11, "bold"), fg="#00E676", bg="#171717", width=24, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=("Helvetica", 10), fg="#CCCCCC", bg="#171717", anchor="w").pack(side="left")

        # -------------------------------------------------------------
        # Section 2: Interactive Keyboard Controls (Rounded Card)
        # -------------------------------------------------------------
        card2 = self._create_card(scrollable_frame, "Interactive Keyboard Controls")
        shortcuts = [
            ("[Space]", "Commit Word", "Pushes active in-progress word into the translated sentence."),
            ("[Backspace]", "Delete", "Removes the last letter from active word, or last word from sentence if word is empty."),
            ("[Escape]", "Clear All", "Immediately resets both active word and sentence buffers."),
            ("[C]", "Calibrate Sign", "Opens the 26-letter calibration modal to record 30 clean frames with 1.5s countdown."),
            ("[T]", "Retrain Models", "Triggers background retraining of the Random Forest / ExtraTrees ensemble."),
        ]
        for key, name, desc in shortcuts:
            row = tk.Frame(card2, bg="#171717")
            row.pack(fill="x", pady=5)
            badge = DarkPillButton(
                row,
                text=key,
                font=("Helvetica", 10, "bold"),
                fg="#00D7FF",
                normal_bg="#202020",
                normal_border="#2E2E2E",
                hover_bg="#282828",
                hover_border="#00D7FF",
                height=26,
                width=100,
                radius=13,
            )
            badge.pack(side="left", padx=(0, 14))
            tk.Label(row, text=name, font=("Helvetica", 10, "bold"), fg="#FFFFFF", bg="#171717", width=14, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=("Helvetica", 10), fg="#AAAAAA", bg="#171717", anchor="w").pack(side="left")

        # -------------------------------------------------------------
        # Section 3: ISL Alphabet Reference (Rounded Card)
        # -------------------------------------------------------------
        card3 = self._create_card(scrollable_frame, "Indian Sign Language (ISL) Alphabet Reference")

        tk.Label(card3, text="One-Hand Signs (8 Letters):", font=("Helvetica", 11, "bold"), fg="#00E676", bg="#171717").pack(anchor="w", pady=(4, 2))
        tk.Label(
            card3,
            text="C (curved hand)  •  I (index up, closed fist)  •  J (sideways tilted hook)  •  L (thumb & index L-shape)\n"
                 "O (closed circular loop)  •  U (pinky up / dual-split)  •  V (index & middle V-shape)  •  W (3 fingers up)",
            font=("Helvetica", 10),
            fg="#CCCCCC",
            bg="#171717",
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(card3, text="Two-Hand Signs (19 Letters):", font=("Helvetica", 11, "bold"), fg="#00D7FF", bg="#171717").pack(anchor="w", pady=(4, 2))
        tk.Label(
            card3,
            text="A (index touches non-dominant thumb)  •  B (dual O-loops)  •  D (index touches vertical thumb)\n"
                 "E (index touches non-dominant index)  •  F (index crossing non-dominant index)  •  G (clenched fists touching)\n"
                 "H (horizontal open palm strike)  •  K (index hooked onto palm)  •  M (3 fingers draped on palm)\n"
                 "N (2 fingers draped on palm)  •  P (loop touching vertical index)  •  Q (thumb & index loop hooked)\n"
                 "R (crossed index on palm)  •  S (pinky hooking pinky)  •  T (index touching palm base)\n"
                 "X (crossed index fingers)  •  Y (thumb & pinky extended on palm)  •  Z (dual palm angle)",
            font=("Helvetica", 10),
            fg="#CCCCCC",
            bg="#171717",
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        # -------------------------------------------------------------
        # Section 4: Accuracy Tips (Rounded Card)
        # -------------------------------------------------------------
        card4 = self._create_card(scrollable_frame, "Tips for Best Recognition Accuracy")
        tips = [
            "• Distance: Position hands 30cm – 60cm in front of webcam with both hands clearly in frame.",
            "• Lighting: Ensure front-facing lighting without strong backlighting or glare on camera lens.",
            "• Calibration: If a specific sign is struggling, press [C], select the letter, and hold steady during 30 frames.",
            "• Duplicate letters: Holding a letter appends once; after a 1.2s cooldown, it will append again.",
            "• Translation: Click the Globe icon (🌐) on Home to instantly switch target language and accent voice.",
        ]
        for tip in tips:
            tk.Label(card4, text=tip, font=("Helvetica", 10), fg="#CCCCCC", bg="#171717", anchor="w").pack(anchor="w", pady=2)

    def _create_card(self, parent, title):
        """Constructs an anti-aliased rounded container card with dark border and auto height."""
        card = DarkRoundedCard(
            parent,
            width=1080,
            height=140,
            radius=20,
            card_bg="#171717",
            border_color="#262626",
            pad=18,
            auto_height=True,
        )
        card.pack(fill="x", pady=8)

        tk.Label(
            card.body,
            text=title,
            font=("Helvetica", 13, "bold"),
            fg="#FFFFFF",
            bg="#171717",
        ).pack(anchor="w", pady=(0, 10))

        return card.body
