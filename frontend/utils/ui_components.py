"""
Custom Pure-Tkinter Dark UI Components for SignSpeak.
Eliminates native macOS Aqua white button chrome by using PIL-backed Label controls.
Renders smooth anti-aliased rounded pill capsules and circular buttons matching the Figma design.
"""
import os
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# Resolve path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from utils.icon_loader import load_icon_pil


class DarkPillButton(tk.Label):
    """
    A fully dark-themed pill capsule button with rounded ends.
    Uses PIL to render anti-aliased pill backgrounds.
    Completely avoids macOS native NSButton light chrome.
    """
    def __init__(
        self,
        parent,
        text="",
        command=None,
        height=40,
        width=None,
        fg="#E6E6E6",
        font=("Helvetica", 11, "bold"),
        normal_bg="#1C1C1C",
        normal_border="#2C2C2C",
        hover_bg="#262626",
        hover_border="#383838",
        active_bg="#141414",
        active_border="#00E676",
        radius=None,
        compound="center",
        padx=12,
        pady=0,
        **kwargs
    ):
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else "#121212"
        # Extract any standard Tk parameters that shouldn't conflict
        kwargs.pop("bg", None)
        kwargs.pop("relief", None)
        kwargs.pop("bd", None)
        kwargs.pop("cursor", None)

        super().__init__(
            parent,
            text=text,
            font=font,
            fg=fg,
            compound=compound,
            bg=parent_bg,
            bd=0,
            cursor="hand2",
            padx=padx,
            pady=pady,
            **kwargs
        )
        self.btn_text = text
        self.command = command
        self.btn_height = height
        self.fixed_width = width
        self.fg = fg
        self.btn_font = font
        self.normal_bg = normal_bg
        self.normal_border = normal_border
        self.hover_bg = hover_bg
        self.hover_border = hover_border
        self.active_bg = active_bg
        self.active_border = active_border
        self.radius = radius or (height // 2)
        self.compound = compound

        self._last_w = 0
        self._normal_photo = None
        self._hover_photo = None
        self._active_photo = None
        self._is_hovered = False
        self._is_pressed = False

        if self.fixed_width:
            self._render_fixed()
        else:
            self.bind("<Configure>", self._on_configure)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _render_pill(self, w, h, fill, border):
        scale = 3
        sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
        r = min(int(self.radius * scale), sh // 2, sw // 2)
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bw = 3  # 1px border at 3x scale
        draw.rounded_rectangle(
            (bw // 2, bw // 2, sw - 1 - bw // 2, sh - 1 - bw // 2),
            radius=r,
            fill=fill,
            outline=border,
            width=bw,
        )
        resized = img.resize((w, h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized, master=self.winfo_toplevel())

    def _render_fixed(self):
        w = self.fixed_width
        h = self.btn_height
        self._normal_photo = self._render_pill(w, h, self.normal_bg, self.normal_border)
        self._hover_photo = self._render_pill(w, h, self.hover_bg, self.hover_border)
        self._active_photo = self._render_pill(w, h, self.active_bg, self.active_border)
        super().configure(image=self._normal_photo)

    def _on_configure(self, event):
        w = event.width
        h = self.btn_height
        if w < 10 or w == self._last_w:
            return
        self._last_w = w
        self._normal_photo = self._render_pill(w, h, self.normal_bg, self.normal_border)
        self._hover_photo = self._render_pill(w, h, self.hover_bg, self.hover_border)
        self._active_photo = self._render_pill(w, h, self.active_bg, self.active_border)
        cur = self._hover_photo if self._is_hovered else self._normal_photo
        super().configure(image=cur)

    def _on_enter(self, e):
        self._is_hovered = True
        if self._hover_photo:
            super().configure(image=self._hover_photo)

    def _on_leave(self, e):
        self._is_hovered = False
        self._is_pressed = False
        if self._normal_photo:
            super().configure(image=self._normal_photo)

    def _on_press(self, e):
        self._is_pressed = True
        if self._active_photo:
            super().configure(image=self._active_photo)

    def _on_release(self, e):
        if self._is_pressed:
            self._is_pressed = False
            cur = self._hover_photo if self._is_hovered else self._normal_photo
            if cur:
                super().configure(image=cur)
            if 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
                if self.command:
                    self.command()

    def set_text(self, new_text, new_fg=None):
        self.btn_text = new_text
        if new_fg:
            self.fg = new_fg
            super().configure(text=new_text, fg=new_fg)
        else:
            super().configure(text=new_text)

    def set_style(self, normal_bg=None, normal_border=None, fg=None, hover_bg=None, hover_border=None):
        if normal_bg:
            self.normal_bg = normal_bg
        if normal_border:
            self.normal_border = normal_border
        if hover_bg:
            self.hover_bg = hover_bg
        if hover_border:
            self.hover_border = hover_border
        if fg:
            self.fg = fg
            super().configure(fg=fg)
        w = self.winfo_width() if not self.fixed_width else self.fixed_width
        if w > 10:
            self._normal_photo = self._render_pill(w, self.btn_height, self.normal_bg, self.normal_border)
            self._hover_photo = self._render_pill(w, self.btn_height, self.hover_bg, self.hover_border)
            self._active_photo = self._render_pill(w, self.btn_height, self.active_bg, self.active_border)
            super().configure(image=self._normal_photo)

    def configure(self, cnf=None, **kwargs):
        """Override configure to intercept style and text updates cleanly."""
        if cnf is not None:
            kwargs.update(cnf)
        text = kwargs.pop("text", None)
        fg = kwargs.pop("fg", None) or kwargs.pop("foreground", None)
        bg = kwargs.pop("bg", None) or kwargs.pop("background", None)
        border = kwargs.pop("highlightbackground", None)
        cmd = kwargs.pop("command", None)
        kwargs.pop("relief", None)
        kwargs.pop("bd", None)

        if cmd is not None:
            self.command = cmd
        if text is not None:
            self.btn_text = text
            kwargs["text"] = text
        if fg is not None:
            self.fg = fg
            kwargs["fg"] = fg
        if bg is not None or border is not None:
            if bg:
                self.normal_bg = bg
            if border:
                self.normal_border = border
            w = self.winfo_width() if not self.fixed_width else self.fixed_width
            if w > 10:
                self._normal_photo = self._render_pill(w, self.btn_height, self.normal_bg, self.normal_border)
                self._hover_photo = self._render_pill(w, self.btn_height, self.hover_bg, self.hover_border)
                kwargs["image"] = self._normal_photo

        if kwargs:
            super().configure(**kwargs)


class DarkCircleButton(tk.Label):
    """
    A circular dark icon button.
    Renders an anti-aliased circular background with centered outline icon.
    Never produces white Aqua borders or square background on macOS.
    """
    def __init__(
        self,
        parent,
        icon_name=None,
        icon_color="777777",
        icon_size=24,
        size=52,
        command=None,
        normal_bg="#181818",
        normal_border="#282828",
        hover_bg="#242424",
        hover_border="#383838",
        active_border="#00E676",
        active_bg="#1A261C",
        **kwargs
    ):
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else "#121212"
        kwargs.pop("bg", None)
        kwargs.pop("relief", None)
        kwargs.pop("bd", None)
        kwargs.pop("cursor", None)

        super().__init__(
            parent,
            bg=parent_bg,
            bd=0,
            cursor="hand2",
            **kwargs
        )
        self.command = command
        self.btn_size = size
        self.icon_name = icon_name
        self.icon_color = icon_color
        self.icon_size = icon_size
        self.normal_bg = normal_bg
        self.normal_border = normal_border
        self.hover_bg = hover_bg
        self.hover_border = hover_border
        self.active_border = active_border
        self.active_bg = active_bg
        self.is_active = False
        self._is_hovered = False

        self._normal_photo = None
        self._hover_photo = None
        self._active_photo = None

        self._rebuild_images()
        super().configure(image=self._normal_photo)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)

    def _render_circle(self, bg_color, border_color, icon_col):
        scale = 3
        s = self.btn_size * scale
        base = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(base)
        bw = 3  # 1px border at 3x scale
        draw.ellipse(
            (bw // 2, bw // 2, s - 1 - bw // 2, s - 1 - bw // 2),
            fill=bg_color,
            outline=border_color,
            width=bw,
        )
        resized = base.resize((self.btn_size, self.btn_size), Image.Resampling.LANCZOS)

        if self.icon_name:
            try:
                icon_img = load_icon_pil(self.icon_name, color=icon_col, size=self.icon_size)
                iw, ih = icon_img.size
                px = (self.btn_size - iw) // 2
                py = (self.btn_size - ih) // 2
                resized.paste(icon_img, (px, py), icon_img)
            except Exception as e:
                print(f"Error drawing icon {self.icon_name}: {e}")

        return ImageTk.PhotoImage(resized, master=self.winfo_toplevel())

    def _rebuild_images(self):
        col = self.icon_color
        self._normal_photo = self._render_circle(self.normal_bg, self.normal_border, col)
        self._hover_photo = self._render_circle(self.hover_bg, self.hover_border, "FFFFFF")
        self._active_photo = self._render_circle(self.active_bg, self.active_border, "00E676")

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            super().configure(image=self._active_photo)
        else:
            super().configure(image=self._hover_photo if self._is_hovered else self._normal_photo)

    def set_icon(self, icon_name, icon_color=None):
        self.icon_name = icon_name
        if icon_color:
            self.icon_color = icon_color
        self._rebuild_images()
        cur = self._active_photo if self.is_active else (self._hover_photo if self._is_hovered else self._normal_photo)
        super().configure(image=cur)

    def configure(self, cnf=None, **kwargs):
        if cnf is not None:
            kwargs.update(cnf)
        cmd = kwargs.pop("command", None)
        if cmd is not None:
            self.command = cmd
        kwargs.pop("relief", None)
        kwargs.pop("bd", None)
        kwargs.pop("bg", None)
        kwargs.pop("highlightbackground", None)
        kwargs.pop("image", None)
        if kwargs:
            super().configure(**kwargs)

    def _on_enter(self, e):
        self._is_hovered = True
        if not self.is_active and self._hover_photo:
            super().configure(image=self._hover_photo)

    def _on_leave(self, e):
        self._is_hovered = False
        if not self.is_active and self._normal_photo:
            super().configure(image=self._normal_photo)

    def _on_click(self, e):
        if 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            if self.command:
                self.command()


class DarkRoundedCard(tk.Canvas):
    """
    A dark container card with anti-aliased rounded corners and subtle border.
    Child widgets are placed inside `self.body`.
    Eliminates all sharp 90-degree rectangular card borders across the UI.
    Supports auto_height=True to automatically expand to fit inner body content.
    """
    def __init__(
        self,
        parent,
        width=300,
        height=200,
        radius=24,
        card_bg="#171717",
        border_color="#262626",
        border_width=1,
        pad=14,
        auto_height=False,
        **kwargs
    ):
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else "#121212"
        kwargs.pop("bg", None)
        kwargs.pop("highlightthickness", None)
        kwargs.pop("bd", None)
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent_bg,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.w = max(10, width)
        self.h = max(10, height)
        self.radius = radius
        self.card_bg = card_bg
        self.border_color = border_color
        self.border_width = border_width
        self.pad = pad
        self.auto_height = auto_height
        self.parent_bg = parent_bg
        self.bg_photo = None
        self._content_win = None

        self.body = tk.Frame(self, bg=card_bg)
        self._render_card(self.w, self.h)
        self.bind("<Configure>", self._on_configure)
        if self.auto_height:
            self.body.bind("<Configure>", self._on_body_configure)

    def _render_card(self, w, h):
        scale = 3
        sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
        sr = min(int(self.radius * scale), sw // 2, sh // 2)
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bw = max(2, int(self.border_width * scale))
        draw.rounded_rectangle(
            (bw // 2, bw // 2, sw - 1 - bw // 2, sh - 1 - bw // 2),
            radius=sr,
            fill=self.card_bg,
            outline=self.border_color,
            width=bw,
        )
        resized = img.resize((w, h), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized, master=self.winfo_toplevel())

        self.delete("card_bg")
        self.create_image(0, 0, image=self.bg_photo, anchor="nw", tags="card_bg")
        self.tag_lower("card_bg")

        bw_val = max(10, w - 2 * self.pad)
        bh_val = max(10, h - 2 * self.pad)
        if self._content_win is None:
            if self.auto_height:
                self._content_win = self.create_window(
                    (self.pad, self.pad),
                    window=self.body,
                    anchor="nw",
                    width=bw_val,
                )
            else:
                self._content_win = self.create_window(
                    (self.pad, self.pad),
                    window=self.body,
                    anchor="nw",
                    width=bw_val,
                    height=bh_val,
                )
        else:
            self.coords(self._content_win, self.pad, self.pad)
            if self.auto_height:
                self.itemconfigure(self._content_win, width=bw_val)
            else:
                self.itemconfigure(self._content_win, width=bw_val, height=bh_val)

    def _on_body_configure(self, event):
        if not self.auto_height:
            return
        needed_h = self.body.winfo_reqheight() + 2 * self.pad
        if needed_h > 20 and abs(needed_h - self.h) > 2:
            self.h = needed_h
            self.configure(height=needed_h)
            self._render_card(self.w, self.h)

    def _on_configure(self, event):
        new_w, new_h = event.width, event.height
        if new_w < 10 or new_h < 10:
            return
        if self.auto_height:
            if abs(new_w - self.w) > 2:
                self.w = new_w
                self._render_card(self.w, self.h)
        else:
            if abs(new_w - self.w) > 2 or abs(new_h - self.h) > 2:
                self.w, self.h = new_w, new_h
                self._render_card(self.w, self.h)


class DarkPillFrame(tk.Canvas):
    """
    A capsule pill container with fully rounded ends.
    Useful for status pills, progress indicators, and compact badges.
    """
    def __init__(
        self,
        parent,
        width=140,
        height=32,
        card_bg="#1E1E1E",
        border_color="#2C2C2C",
        pad_x=8,
        pad_y=2,
        **kwargs
    ):
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else "#181818"
        kwargs.pop("bg", None)
        kwargs.pop("highlightthickness", None)
        kwargs.pop("bd", None)
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent_bg,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.w = max(10, width)
        self.h = max(10, height)
        self.card_bg = card_bg
        self.border_color = border_color
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.bg_photo = None
        self._content_win = None

        self.body = tk.Frame(self, bg=card_bg)
        self._render_pill(self.w, self.h)
        self.bind("<Configure>", self._on_configure)

    def _render_pill(self, w, h):
        scale = 3
        sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
        r = sh // 2
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bw = 3
        draw.rounded_rectangle(
            (bw // 2, bw // 2, sw - 1 - bw // 2, sh - 1 - bw // 2),
            radius=r,
            fill=self.card_bg,
            outline=self.border_color,
            width=bw,
        )
        resized = img.resize((w, h), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized, master=self.winfo_toplevel())

        self.delete("pill_bg")
        self.create_image(0, 0, image=self.bg_photo, anchor="nw", tags="pill_bg")
        self.tag_lower("pill_bg")

        bw_val = max(10, w - 2 * self.pad_x)
        bh_val = max(10, h - 2 * self.pad_y)
        if self._content_win is None:
            self._content_win = self.create_window(
                (self.pad_x, self.pad_y),
                window=self.body,
                anchor="nw",
                width=bw_val,
                height=bh_val,
            )
        else:
            self.coords(self._content_win, self.pad_x, self.pad_y)
            self.itemconfigure(self._content_win, width=bw_val, height=bh_val)

    def _on_configure(self, event):
        new_w, new_h = event.width, event.height
        if new_w < 10 or new_h < 10:
            return
        if abs(new_w - self.w) > 2 or abs(new_h - self.h) > 2:
            self.w, self.h = new_w, new_h
            self._render_pill(self.w, self.h)


