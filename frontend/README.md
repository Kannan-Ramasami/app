# SignSpeak Frontend

This is the frontend UI/UX for the SignSpeak project (Indian Sign Language to Natural Text translation).
It is built entirely using Python standard `tkinter` and `PIL.ImageTk` to provide a modern, high-performance dark-themed GUI (`#121212` main background, `#181818` panels, `#242424` borders) with integrated real-time ML hand-sign recognition.

## Project Structure

```
frontend/
│
├── main.py                # Main application entry point & page router (tk.Tk)
├── README.md              # Frontend documentation
├── assets/                # Cached icons & graphics
├── utils/
│   └── icon_loader.py     # Thread-safe PhotoImage loader & cache
└── pages/                 # UI Pages for the application
    ├── __init__.py
    ├── home.py            # Main ISL translation interface (Camera, HUD, Words, Sentence)
    ├── settings.py        # Settings page (Flat dark styling)
    ├── help.py            # Help page (Flat dark styling)
    └── not_found.py       # Generic 404 placeholder for unlinked buttons
```

## Setup & Requirements

The frontend relies on standard Python `tkinter` and `pillow`. Ensure dependencies are installed:

```bash
pip install pillow opencv-python mediapipe scikit-learn joblib numpy
```

Run the application from root or inside frontend:
```bash
python frontend/main.py
# or
cd frontend && python main.py
```

## Features & Controls

1. **Integrated ML Inference**:
   - Streams webcam frames using non-blocking Tkinter main-thread loop (`self.after(15, ...)`).
   - Detects 1-hand and 2-hand signs using MediaPipe + ensemble ML model (`SignClassifier`).
   - 8-frame majority smoothing window for stable letter predictions.

2. **Letter -> Word -> Sentence Formation**:
   - **Letter Dwell (1.0s)**: Holding a sign steadily for 1.0s appends the letter to the active word.
   - **Duplicate Cooldown (1.2s)**: Prevents repeated character spamming when holding a sign.
   - **Auto-Space (1.6s)**: When no hands are detected for 1.6s, the word automatically commits to the sentence.
   - **Sentence Formatting**: Automatically capitalizes the first word and shows the in-progress word at the tail.

3. **Interactive & Keyboard Controls**:
   - `[Space]` or `＋ Add Word` button: Commits active word to the sentence.
   - `[Backspace]` or `⌫ Delete` button: Deletes last letter (or last word if word buffer is empty).
   - `[Escape]` or `✕ Clear` button: Clears word and sentence buffers.
   - `[Record]` (◎) button or `[C]` key: Opens 26-letter calibration picker to record 30 clean frames with 1.5s countdown and hot-reloads models.
   - `[T]` key: Triggers model retraining and hot-reloads in the background.

## Components Breakdown

- **Top Bar**: Contains the Logo, Contact button, Help (?), and Settings (⚙).
- **Camera Panel (Left)**: Designed to display the live feed. It's currently filled with a placeholder icon and "Enable Camera To Use".
- **Letters/Words Panel (Top Right)**: Displays real-time detected characters or isolated words.
- **Sentence Panel (Bottom Right)**: Displays the context-aware constructed sentence.
- **Bottom Bar**: Contains the START/STOP toggle, Language selection (🌐), and Record (◎) buttons.
