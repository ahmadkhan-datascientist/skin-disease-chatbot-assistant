"""
guardrail.py — 3-Pass Input Validation (Groq API)
FYP: Skin Disease Chatbot Assistant (BS Data Science)

SWITCHED TO: Groq API (free, same key as chatbot.py)

PASSES:
  Pass 1 — Keyword filter   (free, instant, no API call)
  Pass 2 — Image skin check (is_skin_image in app.py)
  Pass 3 — LLM intent check (Groq, only when Pass 1 is ambiguous)
  Decision engine           → allow / reject + reason

SETUP:
  pip install groq
  Same GROQ_API_KEY as chatbot.py
"""

from groq import Groq

# ─────────────────────────────────────────────
# 1. CONFIGURATION — same key as chatbot.py
# ─────────────────────────────────────────────

GROQ_API_KEY = "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"  

GROQ_MODEL   = "llama-3.1-8b-instant"

if GROQ_API_KEY == "YOUR_GROQ_KEY_HERE":
    raise ValueError(
        "Set your Groq API key in guardrail.py.\n"
        "Get free key at: https://console.groq.com"
    )

_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# 2. KEYWORD LISTS
# ─────────────────────────────────────────────

SKIN_ALLOWLIST = [
    "skin", "rash", "lesion", "mole", "melanoma", "acne", "eczema",
    "psoriasis", "dermatitis", "keratosis", "carcinoma", "bcc", "nevi",
    "dermatofibroma", "vascular", "pigment", "freckle", "wart", "blister",
    "hive", "urticaria", "ringworm", "fungal", "infection", "wound",
    "itching", "itchy", "burning", "redness", "swelling", "bump", "lump",
    "spot", "patch", "scab", "crust", "pus", "bleed", "sore", "painful",
    "dry", "flaky", "scaling", "peeling", "discoloration", "dark spot",
    "growth", "ulcer", "boil", "abscess", "inflammation",
    "arm", "leg", "back", "face", "neck", "hand", "foot", "scalp",
    "chest", "stomach", "shoulder", "elbow", "knee", "finger", "toe",
    "chamri", "jild", "kharish", "khujli", "jalan", "sujan", "daag",
    "nishan", "zakhm", "ghao", "waram", "phoda", "lal", "daane",
    "dermatologist", "doctor", "clinic", "diagnosis", "treatment",
    "medicine", "cream", "ointment", "biopsy", "sunscreen",
    "remedy", "cure", "heal", "symptom", "condition", "disease",
]

TOPIC_BLOCKLIST = [
    "capital", "president", "prime minister", "country", "history",
    "geography", "math", "physics", "chemistry", "calculate", "equation",
    "recipe", "cook", "food", "restaurant", "meal", "ingredient",
    "chicken", "rice", "biryani", "pizza", "cake",
    "cricket", "football", "match", "score", "team", "player",
    "movie", "film", "song", "music", "actor",
    "programming", "code", "javascript", "computer", "phone", "game",
    "weather", "temperature", "rain", "forecast",
    "news", "politics", "election", "government", "war",
    "economy", "stock", "price", "joke", "poem",
]

GREETING_PHRASES = [
    "hello", "hi", "hey", "salam", "assalam", "good morning",
    "good evening", "how are you", "what can you do",
    "help me", "start", "begin",
]

GREETING_RESPONSE = (
    "Hello! I am your Skin Disease Assistant.\n\n"
    "I can help you identify and understand common skin conditions.\n\n"
    "To get started:\n"
    "1. Upload a clear, close-up photo of the affected skin area\n"
    "2. Describe your symptoms in the text box\n"
    "3. Click Analyze Skin\n\n"
    "I support both English and Urdu input. "
    "Please note I am a screening tool only — always consult a qualified "
    "dermatologist for proper diagnosis and treatment."
)

REJECT_MSG = (
    "I can only help with skin disease related questions. "
    "Please describe your skin condition or symptoms."
)

# ─────────────────────────────────────────────
# 3. PASS 1 — KEYWORD FILTER (free, instant)
# ─────────────────────────────────────────────

def _pass1_keyword(text):
    """
    Zero-cost keyword check.
    Returns: "allow" / "block" / "greeting" / "ambiguous"
    """
    t = text.lower().strip()
    if not t:
        return "ambiguous"

    for phrase in GREETING_PHRASES:
        if phrase in t:
            return "greeting"

    has_block = any(w in t for w in TOPIC_BLOCKLIST)
    has_skin  = any(w in t for w in SKIN_ALLOWLIST)

    if has_block and not has_skin:
        return "block"
    if has_skin:
        return "allow"
    return "ambiguous"

# ─────────────────────────────────────────────
# 4. PASS 3 — GROQ LLM CHECK
# Only called when Pass 1 is ambiguous
# max_tokens=3 — only YES or NO needed — near zero cost
# ─────────────────────────────────────────────

def _pass3_groq(text):
    """
    Ask Groq LLaMA: is this skin-related? YES or NO.
    Called only when keyword filter is inconclusive.
    Returns: "allow" or "block"
    """
    try:
        response = _client.chat.completions.create(
            model      = GROQ_MODEL,
            max_tokens = 3,
            temperature = 0.0,
            messages   = [
                {
                    "role": "system",
                    "content": (
                        "You are a content filter for a skin disease chatbot. "
                        "Reply with exactly one word: YES if the message is about "
                        "skin conditions, dermatology, or skin health. NO otherwise."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        answer = response.choices[0].message.content.strip().upper()
        return "allow" if "YES" in answer else "block"
    except Exception as e:
        print(f"[guardrail.py] Groq check error: {e}")
        return "allow"   # fail open — chatbot handles edge cases

# ─────────────────────────────────────────────
# 5. DECISION ENGINE
# ─────────────────────────────────────────────

def _decision(pass1_result, text):
    if pass1_result == "greeting":
        return True, "greeting"
    if pass1_result == "allow":
        return True, ""
    if pass1_result == "block":
        return False, REJECT_MSG

    # Ambiguous → Pass 3
    print(f"[guardrail] Ambiguous → Groq check: '{text[:60]}'")
    pass3 = _pass3_groq(text)
    return (True, "") if pass3 == "allow" else (False, REJECT_MSG)

# ─────────────────────────────────────────────
# 6. PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def check_text(text):
    """
    Validate symptom/question text before running pipeline.
    Returns: (allowed: bool, reason: str)
      reason = ""          if allowed
      reason = "greeting"  if greeting detected
      reason = reject msg  if blocked
    """
    if not text or not text.strip():
        return True, ""
    p1 = _pass1_keyword(text)
    return _decision(p1, text)


def check_question(text, disease_label=None, confidence=None):
    """
    Validate follow-up messages. More lenient in active session.
    Returns: (allowed: bool, reason: str)
    """
    if not text or not text.strip():
        return True, ""

    t = text.lower().strip()

    # Hard block — never allow these
    hard_blocks = [
        "cricket", "football", "recipe", "food", "cook",
        "weather", "news", "politics", "movie", "song",
        "capital of", "president of", "who won",
    ]
    if any(b in t for b in hard_blocks):
        return False, (
            "I can only answer questions about your skin condition. "
            "Please ask about symptoms, treatment, or when to see a doctor."
        )

    # Medical/health context — always allow in active session
    medical_words = [
        "doctor", "hospital", "medicine", "cream", "treatment", "cure",
        "serious", "dangerous", "advice", "recommend", "should i",
        "what should", "how long", "when to", "can i", "is it",
        "will it", "does it", "home remedy", "sunscreen", "avoid",
        "prevent", "spread", "contagious", "ok", "thanks", "okay",
        "what about", "tell me more", "explain",
    ]
    if any(w in t for w in medical_words):
        return True, ""

    # Short messages are always fine
    if len(t.split()) <= 5:
        return True, ""

    return check_text(text)


# ─────────────────────────────────────────────
# 7. TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("I have a dark itchy spot on my arm",  True),
        ("mujhe haath par kharish ho rahi hai",  True),
        ("What is the capital of France?",       False),
        ("Who won the cricket match?",           False),
        ("What is the best biryani recipe?",     False),
        ("Should I apply sunscreen?",            True),
        ("Hello",                                True),
        ("Is my condition serious?",             True),
        ("The weather is nice today",            False),
        ("I have a red rash on my leg",          True),
    ]

    print("=" * 55)
    print("GUARDRAIL TESTS — Pass 1 + Pass 3 (Groq)")
    print("=" * 55)

    passed = 0
    for text, expected in cases:
        allowed, reason = check_text(text)
        correct = allowed == expected
        if correct: passed += 1
        mark   = "OK   " if correct else "WRONG"
        status = "ALLOW" if allowed else "BLOCK"
        print(f"[{mark}] {status} | {text[:50]}")

    print(f"\n{passed}/{len(cases)} tests passed.")
