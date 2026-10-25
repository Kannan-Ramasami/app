import os
import urllib.request
import tkinter as tk
from PIL import Image, ImageTk

# Module-level caches
_ICON_CACHE: dict[tuple[int, str, str, int], ImageTk.PhotoImage] = {}
_PIL_CACHE: dict[tuple[str, str, int], Image.Image] = {}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def recolor_icon(icon_img: Image.Image, hex_color: str) -> Image.Image:
    """Recolors an RGBA mask icon to any hex color instantly in memory without network."""
    color_code = hex_color.lstrip("#")
    r, g, b = tuple(int(color_code[i:i+2], 16) for i in (0, 2, 4))
    r_ch, g_ch, b_ch, a_ch = icon_img.split()
    colored = Image.new("RGBA", icon_img.size, (r, g, b, 255))
    colored.putalpha(a_ch)
    return colored


def load_icon_pil(name: str, color: str = "777777", size: int = 48) -> Image.Image:
    """
    Loads an icon from local assets, resizes to (size, size), and tints to the desired color.
    Operates 100% offline using existing local assets.
    """
    color_code = color.lstrip("#").lower()
    pil_key = (name, color_code, size)
    if pil_key in _PIL_CACHE:
        return _PIL_CACHE[pil_key].copy()

    # Look for exact file or any base 777777 file for this icon
    exact_filename = f"{name}_{color_code}_{size * 2}.png"
    exact_path = os.path.join(ASSETS_DIR, exact_filename)

    base_path = None
    if os.path.exists(exact_path):
        base_path = exact_path
    else:
        # Search for any asset starting with the icon name
        for fname in os.listdir(ASSETS_DIR):
            if fname.startswith(f"{name}_") and fname.endswith(".png"):
                base_path = os.path.join(ASSETS_DIR, fname)
                break

    if base_path and os.path.exists(base_path):
        try:
            img = Image.open(base_path).convert("RGBA")
            if img.size != (size, size):
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            if color_code != "777777":
                img = recolor_icon(img, color_code)
            _PIL_CACHE[pil_key] = img
            return img.copy()
        except Exception as e:
            print(f"Error loading base icon '{base_path}': {e}")

    # Fallback transparent placeholder if icon is not found
    img = Image.new("RGBA", (size, size), color=(0, 0, 0, 0))
    _PIL_CACHE[pil_key] = img
    return img.copy()


def load_icon(name: str, color: str = "777777", size: int = 48, master: tk.Misc | None = None) -> ImageTk.PhotoImage:
    """
    Loads an icon and returns a PIL ImageTk.PhotoImage.
    Resized via Image.Resampling.LANCZOS to (size, size).
    Caches loaded PhotoImage references in a module-level dictionary so tkinter never garbage-collects icons.
    """
    color_code = color.lstrip("#").lower()
    active_master = master or getattr(tk, "_default_root", None)
    master_id = id(active_master) if active_master is not None else 0
    cache_key = (master_id, name, color_code, size)

    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    img = load_icon_pil(name, color, size)
    photo = ImageTk.PhotoImage(img, master=active_master)
    _ICON_CACHE[cache_key] = photo
    return photo
