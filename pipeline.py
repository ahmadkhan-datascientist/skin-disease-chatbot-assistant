"""
pipeline.py — Main Series Pipeline
FYP: Skin Disease Chatbot Assistant (BS Data Science)

LANGUAGE HANDLING:
  - English input        → Groq replies in English directly
  - Roman Urdu input      → Groq replies in Roman Urdu directly (no translation library needed)
  - Native Urdu script    → Groq replies in English internally, then translated to Urdu script

SERIES FLOW:
  User input (image + text)
      ↓
  guardrail.py  — keyword filter + LLM check
      ↓
  translation.py — detect language only (en / ur / roman_urdu)
      ↓
  train_skin.py  — CV model predicts disease
      ↓
  chatbot.py     — Groq generates response IN THE DETECTED LANGUAGE
      ↓
  translation.py — only used for native Urdu script translation
      ↓
  Response to user
"""

import os

from chatbot     import chatbot, followup, general_skin_chat, new_conversation, DISEASE_INFO
from translation import translate_pipeline, translate_response, get_language_name
from guardrail   import check_text, check_question, GREETING_RESPONSE
from train_skin  import load_model, predict_disease

# ─────────────────────────────────────────────
# DETECTION BLOCK BUILDER
# ─────────────────────────────────────────────

def build_detection_block(label, confidence, uncertain):
    if uncertain or label is None:
        return None
    info = DISEASE_INFO.get(label, {
        "name":    label,
        "urgency": "MEDIUM",
        "note":    "Please consult a dermatologist for more information.",
    })
    return (
        f"Detected condition : {info['name']}\n"
        f"Confidence         : {confidence:.1f}%\n"
        f"Urgency level      : {info['urgency']}\n"
        f"Medical note       : {info['note']}"
    )

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH = r"D:\Fyp Project\skin_model_best.pth"
_cv_model  = None

# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────

def initialize(model_path=MODEL_PATH):
    global _cv_model
    if not os.path.exists(model_path):
        print(f"[pipeline] ERROR: Model not found at '{model_path}'")
        return False
    try:
        print(f"[pipeline] Loading CV model from '{model_path}'...")
        _cv_model = load_model(model_path)
        print("[pipeline] CV model loaded.")
        return True
    except Exception as e:
        print(f"[pipeline] ERROR loading model: {e}")
        return False

# ─────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────

def new_session():
    return {
        "history":       new_conversation(),
        "disease_label": None,
        "confidence":    None,
        "uncertain":     False,
        "original_lang": "en",
    }

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def full_pipeline(image_input, symptom_text, session=None):
    if session is None:
        session = new_session()

    if _cv_model is None:
        return {"success": False, "error": "CV model not loaded."}, session

    if image_input is None:
        return {"success": False, "error": "No image provided."}, session

    symptom_text = symptom_text.strip() if symptom_text else ""

    # ── Guardrail: check symptom text ─────────────────────────────────
    allowed, reason = check_text(symptom_text)

    if not allowed:
        return {
            "success":          False,
            "error":            reason,
            "disease_label":    None,
            "disease_name":     None,
            "confidence":       None,
            "uncertain":        False,
            "all_probs":        {},
            "chatbot_response": reason,
            "detected_lang":    "en",
            "language_name":    "English",
        }, session

    if reason == "greeting":
        return {
            "success":          True,
            "disease_label":    None,
            "disease_name":     None,
            "confidence":       None,
            "uncertain":        False,
            "all_probs":        {},
            "chatbot_response": GREETING_RESPONSE,
            "detected_lang":    "en",
            "language_name":    "English",
            "error":            None,
        }, session

    # ── Detect language (English / Urdu script / Roman Urdu) ──────────
    # NOTE: We only need english_symptoms for symptom understanding by
    # the CV model context. Groq itself handles Roman Urdu / Urdu input
    # directly without needing translation, since LLaMA understands both.
    try:
        english_symptoms, original_lang = translate_pipeline(symptom_text)
        session["original_lang"] = original_lang
    except Exception as e:
        print(f"[pipeline] Translation error: {e}")
        english_symptoms = symptom_text
        original_lang    = "en"

    # ── CV prediction ─────────────────────────────────────────────────
    try:
        cv = predict_disease(image_input, _cv_model)
    except Exception as e:
        print(f"[pipeline] CV error: {e}")
        return {"success": False, "error": f"Could not analyze image: {e}"}, session

    # ── Save to session ───────────────────────────────────────────────
    session["disease_label"] = cv["label"]
    session["confidence"]    = cv["confidence"]
    session["uncertain"]     = cv["uncertain"]

    # ── Chatbot — pass original (untranslated) text + language ───────
    # We pass the ORIGINAL symptom_text (not the English-translated version)
    # so Groq sees the user's actual Roman Urdu / Urdu / English wording
    # and replies in that same language directly.
    try:
        reply_text, updated_history = chatbot(
            disease_label  = cv["label"],
            confidence     = cv["confidence"],
            uncertain      = cv["uncertain"],
            symptoms       = symptom_text,        # original text, not translated
            history        = session["history"],
            input_language = original_lang,
        )
        session["history"] = updated_history
    except Exception as e:
        print(f"[pipeline] Chatbot error: {e}")
        reply_text = "I encountered an issue. Please consult a dermatologist."

    # ── Only translate for native Urdu script ─────────────────────────
    # Roman Urdu and English replies come back already correct from Groq.
    # Native Urdu script sometimes needs a translation pass as a safety net.
    final_response = reply_text

    return {
        "success":          True,
        "disease_label":    cv["label"],
        "disease_name":     cv["full_name"],
        "confidence":       cv["confidence"],
        "uncertain":        cv["uncertain"],
        "all_probs":        cv["all_probs"],
        "detection_block":  build_detection_block(
                                cv["label"],
                                cv["confidence"],
                                cv["uncertain"]
                            ),
        "chatbot_response": final_response,
        "detected_lang":    original_lang,
        "language_name":    get_language_name(original_lang),
        "error":            None,
    }, session

# ─────────────────────────────────────────────
# FOLLOW-UP CHAT
# ─────────────────────────────────────────────

def chat_followup(user_message, session):
    if not user_message or not user_message.strip():
        return "Please type a message.", session

    if session.get("disease_label") is None:
        return (
            "Please upload a skin image first before asking follow-up questions.",
            session
        )

    # Guardrail
    allowed, reason = check_question(
        user_message,
        session.get("disease_label"),
        session.get("confidence"),
    )
    if not allowed:
        return reason, session

    # Detect language of this follow-up message
    try:
        _, lang = translate_pipeline(user_message)
        if lang == "en" and session["original_lang"] != "en":
            lang = session["original_lang"]
    except Exception:
        lang = session.get("original_lang", "en")

    # Chatbot — pass original message + detected language
    try:
        reply_text, updated_history = followup(
            user_message   = user_message,   # original text, not translated
            disease_label  = session["disease_label"],
            confidence     = session["confidence"],
            uncertain      = session["uncertain"],
            history        = session["history"],
            input_language = lang,
        )
        session["history"] = updated_history
    except Exception as e:
        reply_text = f"Error: {e}. Please try again."

    return reply_text, session

# ─────────────────────────────────────────────
# CHAT WITHOUT IMAGE
# ─────────────────────────────────────────────

def chat_without_image(user_message, session):
    """
    Handle skin-related questions when no image has been uploaded yet.
    """
    if not user_message or not user_message.strip():
        return "Please type a message.", session

    allowed, reason = check_text(user_message)
    if reason == "greeting":
        return GREETING_RESPONSE, session
    if not allowed:
        return reason, session

    # Detect language
    try:
        _, lang = translate_pipeline(user_message)
        session["original_lang"] = lang
    except Exception:
        lang = "en"

    # Chatbot — pass original message + detected language
    try:
        reply_text, updated_history = general_skin_chat(
            user_message   = user_message,   # original text, not translated
            history        = session["history"],
            input_language = lang,
        )
        session["history"] = updated_history
    except Exception as e:
        reply_text = f"Error: {e}. Please try again."

    return reply_text, session
