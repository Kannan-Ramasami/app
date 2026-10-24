import tkinter as tk

class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#121212")
        self.controller = controller
        
        label = tk.Label(
            self,
            text="Settings",
            font=("Helvetica", 32, "bold"),
            fg="#FFFFFF",
            bg="#121212"
        )
        label.pack(pady=(50, 20))
        
        info = tk.Label(
            self,
            text="404 Placeholder Page - Coming Soon",
            font=("Helvetica", 16),
            fg="#777777",
            bg="#121212"
        )
        info.pack(pady=20)
        
        btn = tk.Button(
            self,
            text="← Back to Home",
            font=("Helvetica", 12, "bold"),
            fg="#E6E6E6",
            bg="#1A1A1A",
            activebackground="#2A2A2A",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#242424",
            cursor="hand2",
            padx=24,
            pady=10,
            command=lambda: controller.show_frame("HomePage")
        )
        btn.pack(pady=20)
        btn.bind("<Enter>", lambda e: btn.configure(bg="#2A2A2A"))
        btn.bind("<Leave>", lambda e: btn.configure(bg="#1A1A1A"))
