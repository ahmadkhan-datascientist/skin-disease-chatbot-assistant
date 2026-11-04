"""
translation.py — Multilingual Translation Module
FYP: Skin Disease Chatbot Assistant (BS Data Science)

SETUP:
  pip install deep-translator langdetect

FUNCTIONS USED IN pipeline.py:
  translate_pipeline(text)              → (english_text, original_lang)
  translate_response(response, lang)    → translated response string
  get_language_name(lang_code)          → human readable name
"""

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# ─────────────────────────────────────────────
# ROMAN URDU KEYWORD FALLBACK
# ─────────────────────────────────────────────

ROMAN_URDU_KEYWORDS = [
    "dard", "kharish", "khujli", "jalan", "sujan", "daag", "nishan",
    "chamri", "jild", "waram", "zakhm", "ghao", "rog", "bimari",
    "mujhe", "meri", "mera", "mere", "hai", "hain", "hua", "hui",
    "kya", "kyun", "kaise", "yeh", "wo", "aur", "lekin",
    "nahi", "nahin", "haan", "ji", "aap", "main", "hum",
    "se", "ko", "ka", "ki", "ke", "par", "mein",
    "lal", "safed", "kala", "peela", "surkh",
    "rang", "phoda", "jaga", "jagah", "bahut", "zyada", "kam",
    "tej", "halka", "purana", "naya", "daane", "khaal",
]

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

def detect_language(text):
    """
    Detect language of input text.
    Returns: "ur", "roman_urdu", "en", or "other"
    """
    if not text or not text.strip():
        return "en"

    text_lower = text.lower().strip()

    try:
        detected = detect(text)
    except LangDetectException:
        detected = "en"

    if detected in ["ur", "ar"]:
        return "ur"

    if detected == "en":
        words   = text_lower.split()
        matches = sum(1 for w in words if w in ROMAN_URDU_KEYWORDS)
        if matches >= 2:
            return "roman_urdu"
        return "en"

    return "other"

# ─────────────────────────────────────────────
# TRANSLATION FUNCTIONS
# ─────────────────────────────────────────────

def to_english(text, source_lang):
    if source_lang == "en":
        return text.strip()
    try:
        src = "ur" if source_lang == "ur" else "auto"
        translated = GoogleTranslator(source=src, target="en").translate(text.strip())
        return translated if translated else text.strip()
    except Exception as e:
        print(f"[translation.py] to_english error: {e}")
        return text.strip()


def to_urdu(text):
    try:
        translated = GoogleTranslator(source="en", target="ur").translate(text.strip())
        return translated if translated else text.strip()
    except Exception as e:
        print(f"[translation.py] to_urdu error: {e}")
        return text.strip()

# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

def translate_pipeline(text):
    """
    Step 1 — call BEFORE chatbot.
    Returns (english_text, original_lang)
    Save original_lang to pass into translate_response later.
    """
    if not text or not text.strip():
        return "No symptoms provided.", "en"

    original_lang = detect_language(text)
    english_text  = to_english(text, original_lang)
    return english_text, original_lang


def translate_response(english_response, original_lang):
    """
    Step 2 — call AFTER chatbot to translate response back.
    If user wrote in English, returns as-is.
    """
    if not english_response or not english_response.strip():
        return english_response

    if original_lang == "en":
        return english_response

    if original_lang in ["ur", "roman_urdu"]:
        return to_urdu(english_response)

    try:
        translated = GoogleTranslator(source="en", target=original_lang).translate(english_response.strip())
        return translated if translated else english_response
    except Exception:
        return english_response


def get_language_name(lang_code):
    return {"en": "English", "ur": "Urdu", "roman_urdu": "Roman Urdu", "other": "Other"}.get(lang_code, "Unknown")


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "I have a dark spot on my arm",
        "mujhe haath par kala daag hai aur jalan ho rahi hai",
        "مجھے ہاتھ پر کالا داغ ہے",
        "meri chamri par sujan aur kharish hai",
    ]
    for text in tests:
        english, lang = translate_pipeline(text)
        response      = translate_response("Please see a dermatologist.", lang)
        print(f"\nInput   : {text}")
        print(f"Lang    : {get_language_name(lang)}")
        print(f"English : {english}")
        print(f"Response: {response}")
