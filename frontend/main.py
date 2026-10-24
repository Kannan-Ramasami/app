import os
import sys
import tkinter as tk

# Resolve paths so backend and frontend modules can be imported seamlessly
# regardless of whether launched from workspace root or inside frontend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [PARENT_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

app_project_path = os.path.join(PARENT_DIR, "app-project")
if os.path.exists(app_project_path) and app_project_path not in sys.path:
    sys.path.insert(0, app_project_path)

from pages.home import HomePage
from pages.settings import SettingsPage
from pages.help import HelpPage
from pages.not_found import NotFoundPage


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("SignSpeak | ISL Real-Time Translation")
        self.geometry("1150x760")
        self.minsize(1000, 680)
        self.configure(bg="#121212")
        
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
