"""
Real-time multi-language translation engine for SignSpeak.
Supports English, Hindi, Tamil, Telugu, Spanish, French, and German.
Includes offline dictionary fallback and online Google Translate API with caching.
"""
import urllib.request
import urllib.parse
import json
import threading

LANG_CODES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

LANG_VOICES: dict[str, str] = {
    "English": "Samantha",
    "Hindi": "Lekha",
    "Tamil": "Vani",
    "Telugu": "Geeta",
    "Spanish": "Mónica",
    "French": "Thomas",
    "German": "Anna",
}

# Offline Dictionary for common ISL letters, words, and phrases
OFFLINE_DICT: dict[str, dict[str, str]] = {
    "HELLO": {
        "Hindi": "नमस्ते",
        "Tamil": "வணக்கம்",
        "Telugu": "నమస్కారం",
        "Spanish": "Hola",
        "French": "Bonjour",
        "German": "Hallo",
    },
    "HI": {
        "Hindi": "नमस्ते",
        "Tamil": "வணக்கம்",
        "Telugu": "హలో",
        "Spanish": "Hola",
        "French": "Salut",
        "German": "Hallo",
    },
    "HOW ARE YOU": {
        "Hindi": "आप कैसे हैं?",
        "Tamil": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "Telugu": "మీరు ఎలా ఉన్నారు?",
        "Spanish": "¿Cómo estás?",
        "French": "Comment allez-vous?",
        "German": "Wie geht es dir?",
    },
    "THANK YOU": {
        "Hindi": "धन्यवाद",
        "Tamil": "நன்றி",
        "Telugu": "ధన్యవాదాలు",
        "Spanish": "Gracias",
        "French": "Merci",
        "German": "Danke",
    },
    "THANKS": {
        "Hindi": "धन्यवाद",
        "Tamil": "நன்றி",
        "Telugu": "ధన్యవాదాలు",
        "Spanish": "Gracias",
        "French": "Merci",
        "German": "Danke",
    },
    "PLEASE": {
        "Hindi": "कृपया",
        "Tamil": "தயவுசெய்து",
        "Telugu": "దయచేసి",
        "Spanish": "Por favor",
        "French": "S'il vous plaît",
        "German": "Bitte",
    },
    "YES": {
        "Hindi": "हाँ",
        "Tamil": "ஆம்",
        "Telugu": "అవును",
        "Spanish": "Sí",
        "French": "Oui",
        "German": "Ja",
    },
    "NO": {
        "Hindi": "नहीं",
        "Tamil": "இல்லை",
        "Telugu": "కాదు",
        "Spanish": "No",
        "French": "Non",
        "German": "Nein",
    },
    "GOOD": {
        "Hindi": "अच्छा",
        "Tamil": "நல்லது",
        "Telugu": "మంచిది",
        "Spanish": "Bueno",
        "French": "Bien",
        "German": "Gut",
    },
    "MORNING": {
        "Hindi": "सुप्रभात",
        "Tamil": "காலை வணக்கம்",
        "Telugu": "శుభోదయం",
        "Spanish": "Buenos días",
        "French": "Bonjour",
        "German": "Guten Morgen",
    },
    "NIGHT": {
        "Hindi": "शुभ रात्रि",
        "Tamil": "இரவு வணக்கம்",
        "Telugu": "శుభరాత్రి",
        "Spanish": "Buenas noches",
        "French": "Bonne nuit",
        "German": "Gute Nacht",
    },
    "HELP": {
        "Hindi": "मदद",
        "Tamil": "உதவி",
        "Telugu": "సహాయం",
        "Spanish": "Ayuda",
        "French": "Aide",
        "German": "Hilfe",
    },
    "WATER": {
        "Hindi": "पानी",
        "Tamil": "தண்ணீர்",
        "Telugu": "నీరు",
        "Spanish": "Agua",
        "French": "Eau",
        "German": "Wasser",
    },
    "NAME": {
        "Hindi": "नाम",
        "Tamil": "பெயர்",
        "Telugu": "పేరు",
        "Spanish": "Nombre",
        "French": "Nom",
        "German": "Name",
    },
    "WHAT": {
        "Hindi": "क्या",
        "Tamil": "என்ன",
        "Telugu": "ఏమిటి",
        "Spanish": "Qué",
        "French": "Quoi",
        "German": "Was",
    },
    "WHERE": {
        "Hindi": "कहाँ",
        "Tamil": "எங்கே",
        "Telugu": "ఎక్కడ",
        "Spanish": "Dónde",
        "French": "Où",
        "German": "Wo",
    },
    "SIGN": {
        "Hindi": "संकेत",
        "Tamil": "சைகை",
        "Telugu": "సంకేతం",
        "Spanish": "Seña",
        "French": "Signe",
        "German": "Zeichen",
    },
    "SPEAK": {
        "Hindi": "बोलना",
        "Tamil": "பேசு",
        "Telugu": "మాట్లాడు",
        "Spanish": "Hablar",
        "French": "Parler",
        "German": "Sprechen",
    },
}

_CACHE: dict[tuple[str, str], str] = {}


def translate_text(text: str, target_lang: str) -> str:
    """
    Translates text from English to target_lang.
    Checks memory cache -> offline dictionary -> Google Translate endpoint.
    """
    if not text or not text.strip():
        return ""
    
    clean_text = text.strip()
    if target_lang == "English" or target_lang not in LANG_CODES:
        return clean_text

    cache_key = (clean_text.upper(), target_lang)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # Offline dictionary check for exact match
    upper_key = clean_text.upper()
    if upper_key in OFFLINE_DICT and target_lang in OFFLINE_DICT[upper_key]:
        res = OFFLINE_DICT[upper_key][target_lang]
        _CACHE[cache_key] = res
        return res

    # Online Google Translate API
    try:
        code = LANG_CODES[target_lang]
        url = (
            f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={code}&dt=t&q="
            + urllib.parse.quote(clean_text)
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part and part[0]]).strip()
            if translated:
                _CACHE[cache_key] = translated
                return translated
    except Exception:
        pass

    # Offline word-by-word fallback
    words = clean_text.split()
    translated_words = []
    for w in words:
        w_up = w.upper()
        if w_up in OFFLINE_DICT and target_lang in OFFLINE_DICT[w_up]:
            translated_words.append(OFFLINE_DICT[w_up][target_lang])
        else:
            translated_words.append(w)
    res = " ".join(translated_words)
    _CACHE[cache_key] = res
    return res


def translate_async(text: str, target_lang: str, callback) -> None:
    """Translates asynchronously on a background thread and invokes callback(result)."""
    def _worker():
        res = translate_text(text, target_lang)
        callback(res)
    threading.Thread(target=_worker, daemon=True).start()
