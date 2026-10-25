import tkinter as tk
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from utils.ui_components import DarkPillButton


class NotFoundPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller
        
        label = tk.Label(
            self,
            text="404 - Not Found",
            font=("Helvetica", 32, "bold"),
            fg="#FFFFFF",
            bg="#121212"
        )
        label.pack(pady=(50, 20))
        
        info = tk.Label(
            self,
            text="This page has not been implemented yet.",
            font=("Helvetica", 16),
            fg="#777777",
            bg="#121212"
        )
        info.pack(pady=20)
        
        btn = DarkPillButton(
            self,
            text="← Back to Home",
            font=("Helvetica", 12, "bold"),
            fg="#E6E6E6",
            normal_bg="#1A1A1A",
            normal_border="#242424",
            hover_bg="#2A2A2A",
            hover_border="#383838",
            height=44,
            width=160,
            command=lambda: controller.show_frame("HomePage")
        )
        btn.pack(pady=20)

