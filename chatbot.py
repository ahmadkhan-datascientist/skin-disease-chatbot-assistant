"""
chatbot.py — NLP Chatbot Module (Groq API)
FYP: Skin Disease Chatbot Assistant (BS Data Science)

FIXES IN THIS VERSION:
  1. history properly maintained between turns (repeat response bug fix)
  2. general_skin_chat() added — answers skin questions without image
  3. followup() now passes full history so context is maintained
"""

import re
from groq import Groq

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

GROQ_API_KEY = "use your api"
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 300
TEMPERATURE  = 0.6

_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# 2. DISEASE INFO
# ─────────────────────────────────────────────

DISEASE_INFO = {
    "mel":   {"name": "Melanoma",              "urgency": "HIGH",   "note": "Serious skin cancer — needs urgent medical attention."},
    "nv":    {"name": "Melanocytic Nevi",       "urgency": "LOW",    "note": "Usually a harmless mole — monitor for changes."},
    "bcc":   {"name": "Basal Cell Carcinoma",   "urgency": "HIGH",   "note": "Most common skin cancer — treatable when caught early."},
    "akiec": {"name": "Actinic Keratoses",      "urgency": "MEDIUM", "note": "Precancerous condition caused by sun damage."},
    "bkl":   {"name": "Benign Keratosis",       "urgency": "LOW",    "note": "Harmless skin growth — usually no treatment needed."},
    "df":    {"name": "Dermatofibroma",         "urgency": "LOW",    "note": "Common benign skin growth — usually harmless."},
    "vasc":  {"name": "Vascular Lesion",        "urgency": "LOW",    "note": "Blood vessel abnormality — mostly benign."},
}

DISCLAIMER = (
    "\n\nIMPORTANT: This is an AI screening tool only — not a medical diagnosis. "
    "Always consult a qualified dermatologist for proper evaluation and treatment."
)

# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

def new_conversation():
    return []

def clean_response(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'\n{3,}',    '\n\n',    text)
    return text.strip()

def _build_messages(system_prompt, history, user_message):
    """
    FIX 2: Build full message list including ALL previous history.
    This is what was missing — history was not being included properly.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Include ALL previous conversation turns
    for msg in history:
        role = "assistant" if msg["role"] == "bot" else "user"
        # Skip the disclaimer part when sending to API to save tokens
        text = msg["text"]
        if "IMPORTANT: This is an AI screening tool" in text:
            text = text[:text.index("IMPORTANT: This is an AI screening tool")].strip()
        messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_message})
    return messages

# ─────────────────────────────────────────────
# 4. PROMPT BUILDERS
# ─────────────────────────────────────────────

def build_system_prompt(disease_label, confidence, uncertain, input_language="en"):
    """Build prompt based on CV result and user's input language."""

    if input_language == "roman_urdu":
        lang_instruction = (
            "IMPORTANT: The user wrote in Roman Urdu. "
            "You MUST reply in Roman Urdu (Urdu words written in English letters). "
            "Example style: 'Aap ki skin mein yeh masla ho sakta hai...' "
            "Do NOT reply in English. Do NOT reply in native Urdu script. "
            "Only Roman Urdu."
        )
    elif input_language == "ur":
        lang_instruction = (
            "IMPORTANT: The user wrote in Urdu script. "
            "You MUST reply in native Urdu script only. "
            "Do NOT reply in English or Roman Urdu."
        )
    else:
        lang_instruction = "Reply in English."

    if uncertain:
        return (
            f"{lang_instruction} "
            "You are a caring skin health assistant. "
            "Write exactly 3 sentences. No headers. No lists. Plain text only. "
            "Sentence 1: Gently say the image was unclear and ask for a better photo. "
            "Sentence 2: Ask one question about their symptoms. "
            "Sentence 3: Suggest seeing a dermatologist to be safe."
        )

    info = DISEASE_INFO.get(disease_label, {
        "name": disease_label,
        "urgency": "MEDIUM",
        "note": "Please consult a dermatologist.",
    })
    name    = info["name"]
    urgency = info["urgency"]

    urgency_line = {
        "HIGH":   "Please see a dermatologist urgently — within the next 1-2 days, do not delay.",
        "MEDIUM": "Try to book a dermatologist appointment within the next 2-4 weeks.",
        "LOW":    "It is generally harmless, but a routine checkup with a dermatologist is a good idea.",
    }.get(urgency, "Please consult a dermatologist soon.")

    return (
        f"{lang_instruction} "
        f"You are a caring skin health assistant helping a patient screened for {name}. "
        f"You remember the full conversation history. "
        f"Write in plain conversational text. No headers. No bullet points. No markdown. "
        f"Keep response under 80 words. "
        f"For the first message: warm empathetic opening + what {name} is in simple words + "
        f"two home care tips + {urgency_line} "
        f"End with one warm encouraging sentence. Do not use emojis. "
        f"For follow-up messages: directly answer what the user just asked using conversation history."
    )

def build_general_prompt(input_language="en"):
    """Prompt for answering skin questions without any image/CV result."""

    if input_language == "roman_urdu":
        lang_instruction = "IMPORTANT: Reply in Roman Urdu only (Urdu words in English letters)."
    elif input_language == "ur":
        lang_instruction = "IMPORTANT: Reply in native Urdu script only."
    else:
        lang_instruction = "Reply in English."

    return (
        f"{lang_instruction} "
        "You are a helpful skin health assistant chatbot. "
        "You answer general questions about skin conditions, symptoms, skincare, and dermatology. "
        "You ONLY discuss skin-related topics. "
        "If asked about anything unrelated to skin, say: "
        "'I can only help with skin disease questions.' "
        "Write in plain conversational text. No headers. No markdown. Under 100 words. "
        "Be warm and helpful. Always recommend consulting a dermatologist for proper diagnosis. "
        "If the user describes symptoms, give general guidance and suggest uploading a skin photo "
        "for a more accurate AI-based screening."
    )

# ─────────────────────────────────────────────
# 5. GROQ API CALL
# ─────────────────────────────────────────────

def _call_groq(messages):
    """Central Groq API call with error handling."""
    try:
        response = _client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return clean_response(response.choices[0].message.content), None
    except Exception as e:
        err = str(e)
        print(f"[chatbot.py ERROR] {err}")
        if "429" in err or "rate" in err.lower() or "limit" in err.lower():
            return None, "The AI assistant is busy. Please wait 30 seconds and try again."
        elif "401" in err or "invalid" in err.lower():
            return None, "API key issue. Please check GROQ_API_KEY in chatbot.py."
        else:
            return None, f"An error occurred. Please try again. ({err[:60]})"

# ─────────────────────────────────────────────
# 6. DETECTION BLOCK FORMATTER
# ─────────────────────────────────────────────

def _force_format(disease_label, confidence, uncertain, groq_paragraph):
    """Combine fixed detection block + Groq paragraph."""
    if uncertain or disease_label is None:
        return groq_paragraph

    info = DISEASE_INFO.get(disease_label, {
        "name":    disease_label,
        "urgency": "MEDIUM",
        "note":    "Please consult a dermatologist.",
    })
    detection_block = (
        f"Detected condition : {info['name']}\n"
        f"Confidence         : {confidence:.1f}%\n"
        f"Urgency level      : {info['urgency']}\n"
        f"Medical note       : {info['note']}"
    )
    return f"{detection_block}\n\n{groq_paragraph}"

# ─────────────────────────────────────────────
# 7. MAIN CHATBOT FUNCTION
# ─────────────────────────────────────────────

def chatbot(disease_label, confidence, uncertain, symptoms, history=None, input_language="en"):

    """
    Main chatbot — called after CV analysis.

    FIX: history is now properly included in every API call,
    so follow-up messages get different responses based on context.
    """
    if history is None:
        history = []

    system_prompt = build_system_prompt(disease_label, confidence, uncertain, input_language)

    symptom_text  = symptoms.strip() if symptoms and symptoms.strip() else "No symptoms provided."

    messages = _build_messages(system_prompt, history, symptom_text)
    reply_text, error = _call_groq(messages)

    if error:
        reply_text = error

    reply_with_disclaimer = _force_format(disease_label, confidence, uncertain, reply_text) + DISCLAIMER

    # FIX: save both user message AND bot response to history
    updated_history = history + [
        {"role": "user", "text": symptom_text},
        {"role": "bot",  "text": reply_with_disclaimer},
    ]

    return reply_with_disclaimer, updated_history


def followup(user_message, disease_label, confidence, uncertain, history, input_language="en"):

    """
    Handle follow-up questions.

    FIX: passes full history so model knows what was already discussed.
    """
    if history is None:
        history = []

    system_prompt = build_system_prompt(disease_label, confidence, uncertain, input_language)

    messages      = _build_messages(system_prompt, history, user_message)
    reply_text, error = _call_groq(messages)

    if error:
        reply_text = error

    reply_with_disclaimer = reply_text + DISCLAIMER

    # FIX: append this turn to history
    updated_history = history + [
        {"role": "user", "text": user_message},
        {"role": "bot",  "text": reply_with_disclaimer},
    ]

    return reply_with_disclaimer, updated_history


# ─────────────────────────────────────────────
# 8. FIX 3: GENERAL SKIN CHAT (no image needed)
# ─────────────────────────────────────────────

def general_skin_chat(user_message, history=None, input_language="en"):
    """
    Answer general skin questions when no image has been uploaded.

    Instead of blocking the user with "upload an image first",
    this function answers their question and gently suggests
    uploading an image for a more accurate screening.
    """
    if history is None:
        history = []

    system_prompt = build_general_prompt(input_language)
    messages      = _build_messages(system_prompt, history, user_message)
    reply_text, error = _call_groq(messages)

    if error:
        reply_text = error

    reply_with_disclaimer = reply_text + DISCLAIMER

    updated_history = history + [
        {"role": "user", "text": user_message},
        {"role": "bot",  "text": reply_with_disclaimer},
    ]

    return reply_with_disclaimer, updated_history


# ─────────────────────────────────────────────
# 9. TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("TEST 1: Melanoma — first response")
    print("=" * 55)
    h = new_conversation()
    r1, h = chatbot("mel", 87.3, False, "Dark spot on my arm", h)
    print(r1)

    print("\n" + "=" * 55)
    print("TEST 2: Follow-up — should NOT repeat first response")
    print("=" * 55)
    r2, h = followup("Should I go to hospital now?", "mel", 87.3, False, h)
    print(r2)

    print("\n" + "=" * 55)
    print("TEST 3: Second follow-up — should remember full context")
    print("=" * 55)
    r3, h = followup("What cream can I apply at home?", "mel", 87.3, False, h)
    print(r3)

    print("\n" + "=" * 55)
    print("TEST 4: General skin chat — no image uploaded")
    print("=" * 55)
    h4 = new_conversation()
    r4, h4 = general_skin_chat("I have red itchy patches on my arms, what could it be?", h4)
    print(r4)
