import os
import urllib.request
from PIL import Image, ImageTk

# Module-level cache to ensure tkinter never garbage-collects icons
_ICON_CACHE: dict[tuple[str, str, int], ImageTk.PhotoImage] = {}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_icon(name: str, color: str = "777777", size: int = 48) -> ImageTk.PhotoImage:
    """
    Downloads an icon from Icons8 if it doesn't exist, and returns a PIL ImageTk.PhotoImage.
    Resized via Image.Resampling.LANCZOS to (size, size).
    Caches loaded PhotoImage references in a module-level dictionary so tkinter never garbage-collects icons.
    """
    color_code = color.lstrip("#")
    cache_key = (name, color_code.lower(), size)
    
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    img_size = size * 2
    filename = f"{name}_{color_code}_{img_size}.png"
    filepath = os.path.join(ASSETS_DIR, filename)

    if not os.path.exists(filepath):
        url = f"https://img.icons8.com/ios-glyphs/{img_size}/{color_code}/{name}.png"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
        except Exception as e:
            # Fallback transparent placeholder if offline
            print(f"Warning: Failed to fetch icon '{name}': {e}")
            img = Image.new('RGBA', (size, size), color=(0, 0, 0, 0))
            photo = ImageTk.PhotoImage(img)
            _ICON_CACHE[cache_key] = photo
            return photo

    try:
        img = Image.open(filepath).convert("RGBA")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        _ICON_CACHE[cache_key] = photo
        return photo
    except Exception as e:
        print(f"Error loading icon image '{filepath}': {e}")
        img = Image.new('RGBA', (size, size), color=(0, 0, 0, 0))
        photo = ImageTk.PhotoImage(img)
        _ICON_CACHE[cache_key] = photo
        return photo
