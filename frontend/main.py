import os
import sys
import tkinter as tk

# Resolve paths so backend and frontend modules can be imported seamlessly
# regardless of whether launched from workspace root or inside frontend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
REAL_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
REAL_PARENT_DIR = os.path.dirname(REAL_CURRENT_DIR)

for path in [
    PARENT_DIR,
    CURRENT_DIR,
    REAL_PARENT_DIR,
    REAL_CURRENT_DIR,
    os.path.join(PARENT_DIR, "app-project"),
    os.path.join(REAL_PARENT_DIR, "app-project"),
]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from pages.home import HomePage
from pages.settings import SettingsPage
from pages.help import HelpPage
from pages.not_found import NotFoundPage


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("SignSpeak | ISL Real-Time Translation")
        self.geometry("1160x780")
        self.resizable(False, False)
        self.configure(bg="#121212")
        
        # Shared Application Configuration
        self.config = {
            "camera_index": 0,
            "min_detection_confidence": 0.50,
            "min_tracking_confidence": 0.50,
            "dwell_time": 1.0,
            "autospace_pause": 1.6,
            "duplicate_cooldown": 1.2,
            "tts_enabled": False,
            "tts_rate": 175,
            "tts_voice": "Default",
            "active_language": "English",
        }

        # Container for pages
        self.container = tk.Frame(self, bg="#121212")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.pages = {}
        for F in (HomePage, SettingsPage, HelpPage, NotFoundPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame("HomePage")
        
        # Protocol handler for clean shutdown
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def speak_text(self, text: str, lang: str = None):
        """Speaks text asynchronously via background thread using macOS 'say' command."""
        if not text or not text.strip():
            return
        def _worker():
            import subprocess
            from utils.translator import LANG_VOICES
            active_lang = lang or self.config.get("active_language", "English")
            voice = self.config.get("tts_voice", "Default")
            if voice == "Default":
                voice = LANG_VOICES.get(active_lang, "Samantha")
            rate = str(self.config.get("tts_rate", 175))
            cmd = ["/usr/bin/say", "-r", rate]
            if voice:
                cmd.extend(["-v", voice])
            cmd.append(text.strip())
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"TTS execution error: {e}")
        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def get_setting(self, key, default=None):
        return self.config.get(key, default)

    def set_setting(self, key, value):
        self.config[key] = value
        home = self.pages.get("HomePage")
        if home and hasattr(home, "on_settings_updated"):
            home.on_settings_updated()
        
    def show_frame(self, page_name):
        frame = self.pages[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def on_closing(self):
        """Cleanly releases camera, detector, and threads before closing."""
        home_page = self.pages.get("HomePage")
        if home_page and hasattr(home_page, "cleanup"):
            try:
                home_page.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
