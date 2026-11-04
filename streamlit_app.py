"""
streamlit_app.py — Streamlit Frontend
FYP: Skin Disease Chatbot Assistant (BS Data Science)

═══════════════════════════════════════════════════════════════════
MIGRATION NOTE — READ BEFORE EDITING
═══════════════════════════════════════════════════════════════════
This file is the ONLY new file in the migration. It REPLACES:
  - app.py        (Flask routes: /, /analyze, /chat, /reset, /health)
  - index.html    (HTML/CSS/JS chat UI)

It does NOT replace, and does NOT modify, any of the following —
they are imported and called exactly as app.py called them:
  - pipeline.py    (full_pipeline, chat_followup, new_session, initialize)
  - chatbot.py     (Groq LLM logic, disease info, prompts)
  - guardrail.py   (3-pass input validation)
  - translation.py (language detection / translation)
  - train_skin.py  (ResNet-50 load_model / predict_disease)

Every backend call below mirrors the corresponding Flask route 1:1:
  Flask route          →  Streamlit equivalent
  ───────────────────────────────────────────────────────────
  GET  "/"             →  page renders on script run
  POST "/analyze"      →  "Analyze Skin" button handler
  POST "/chat"         →  st.chat_input handler (+ quick-question buttons)
  POST "/reset"        →  "Clear Chat" button handler
  flask.session         →  st.session_state["pipeline_session"]

CAMERA CAPTURE ADDENDUM (frontend-only addition — backend untouched):
  A "Capture Photo" mode was added alongside the existing "Upload Image"
  mode. It uses Streamlit's built-in st.camera_input(), which returns the
  same UploadedFile-shaped object as st.file_uploader() — so it is fed
  into the EXACT same save → validate → full_pipeline() → cleanup flow
  below. No backend file, function signature, or prediction/translation/
  chatbot logic was touched to support this. Retake is implemented by
  resetting the camera widget's `key`, which is the standard Streamlit
  pattern for clearing a widget's value before Analyze is ever clicked.

IMAGE-CONTENT GUARDRAIL ADDENDUM (frontend-only addition — backend
untouched):
  Investigation found that NEITHER the original Flask version NOR the
  Streamlit version ever had a semantic "is this actually skin?" check.
  is_skin_image() below only ever rejected blank/near-black frames; the
  CNN itself is a closed-set 7-class classifier with no "not skin" option,
  so faces/rooms/objects were always silently force-classified into one
  of the 7 trained disease labels. is_relevant_skin_image() below adds a
  real semantic check using Groq's vision API (the same Groq integration
  and API key already used by chatbot.py / guardrail.py), and is called
  AFTER is_skin_image() and BEFORE full_pipeline() — so this is still a
  pure pre-filter, entirely inside this file, with prediction/chatbot/
  translation/guardrail logic completely untouched.

RUN:
  streamlit run streamlit_app.py
═══════════════════════════════════════════════════════════════════
"""

import os
import io
import math
import time
import uuid
import base64
import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq

# requests — Pass C (OpenRouter/Together vision API) ke liye.
# Streamlit ke saath pehle se installed hota hai, koi nayi cheez nahi.
import requests

# OpenCV — LOCAL face detection ke liye (Pass B). Required:
#   pip install opencv-python
try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False

# ── UNCHANGED BACKEND IMPORT — identical to app.py ─────────────────
# This is the single integration point with the existing project.
# No function signatures, return values, or call order are altered.
from pipeline import initialize, full_pipeline, chat_followup, new_session

# ─────────────────────────────────────────────
# APP CONFIGURATION
# (same constants as app.py — copied, not changed)
# ─────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH_MB = 10  # informational; Streamlit's own uploader cap is set below

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Same model path used by pipeline.py / app.py. Left untouched on purpose —
# changing this would be a logic change, which is explicitly out of scope.
MODEL_PATH = r"D:\Fyp Project\skin_model_best.pth"

# ─────────────────────────────────────────────
# SINGLE REJECTION MESSAGE — copied verbatim from app.py
# ─────────────────────────────────────────────

INVALID_IMAGE_MSG = (
    "Please upload a clear, close-up photo of the affected skin area only. "
    "Portraits, selfies, animals, objects, and non-skin images cannot be analyzed."
)

# ─────────────────────────────────────────────
# IMAGE VALIDATOR — copied verbatim from app.py
# (Only rejects blank/solid-color or pitch-black images; everything else
#  passes through to the CV pipeline, exactly as before.)
# ─────────────────────────────────────────────

def is_skin_image(image_path):
    """Unchanged from app.py — only rejects blank/dark images."""
    try:
        img = Image.open(image_path).convert("RGB").resize((100, 100))
        arr = np.array(img, dtype=np.float32)

        if arr.std() < 5:
            return False, INVALID_IMAGE_MSG

        if arr.mean() < 15:
            return False, INVALID_IMAGE_MSG

        return True, ""

    except Exception as e:
        print(f"[streamlit_app.py] Image validation error: {e}")
        return True, ""


def allowed_file(filename):
    """Unchanged from app.py."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# NOTE ON DEAD CODE: app.py also defines contains_face() / _ensure_cascade()
# (a Haar-cascade face detector) but NEVER calls them inside any Flask route.
# They have zero effect on real output, so they are intentionally not ported
# here — porting unused code would add an opencv dependency for no behavioral
# benefit. The NEW is_relevant_skin_image() function below replaces the need
# for that dead code with an actually-wired-in semantic check.


# ─────────────────────────────────────────────
# NEW — IMAGE-CONTENT GUARDRAIL (FULLY LOCAL — no API required)
# Added entirely within this file, per the project constraint that only
# streamlit_app.py may be modified. Conceptually this is the "Pass 2 —
# Image skin check" that guardrail.py's own docstring already reserved a
# slot for, but it is implemented here so that guardrail.py, pipeline.py,
# chatbot.py, translation.py, and train_skin.py all remain untouched.
#
# MIGRATION HISTORY (root-cause investigation, documented for the thesis):
#   v1: Groq "meta-llama/llama-4-scout-17b-16e-instruct" — worked, then
#       Groq RETIRED the model (API: 404 model_not_found). Combined with
#       the old fail-OPEN except block, this silently disabled the filter
#       for two weeks: cars/animals/faces reached the closed-set CNN and
#       received confident disease labels.
#   v2: Groq "qwen/qwen3.6-27b" (Groq's only remaining vision model) —
#       exists on the account but returned 503 "over capacity" on 8+
#       attempts across different times and API keys. Unusable as a
#       reliability-critical safety filter on the free tier.
#   v3: Google Gemini Vision — worked, but free-tier token quota is
#       consumed quickly with image inputs.
#   v4: FULLY LOCAL two-pass filter (Pass A + Pass B below) as the
#       PRIMARY layer — zero API dependency for the safety-critical
#       rejections. NOTE: the ORIGINAL Flask app.py already contained
#       contains_face() as dead code that was never wired in — v4
#       finally activates that idea for real.
#   v5 (CURRENT): Pass C semantic check moved to an OpenAI-compatible
#       vision API — OpenRouter or Together AI (user's choice, same
#       code for both). Pass C answers the question local heuristics
#       cannot: "is a visible skin CONDITION present?" (e.g. rejecting
#       healthy-skin close-ups). If Pass C is unavailable, the app
#       still runs on the local passes — no single provider can kill
#       the filter again.
#
# SETUP (one time):
#   pip install opencv-python        <-- REQUIRED for Pass B (faces)
#   (requests is already installed with Streamlit — nothing else needed)
# ─────────────────────────────────────────────

# ── Pass C provider config — sirf ye 2 lines bharni hain ─────────────
# 1) Provider chuno: "openrouter"  ya  "together"
# 2) Us provider ki API key daalo:
#      OpenRouter -> https://openrouter.ai  -> sign in -> Keys      your-new-key-here
#      Together   -> https://api.together.ai -> sign in -> API Keys
VISION_API_PROVIDER = "openrouter"          # "openrouter" ya "together"
VISION_API_KEY      = "your-new-key-here"      # <-- YAHAN KEY DAALO

# Model lineups change over time (the scout lesson!). Each provider
# gets a fallback LIST — the code tries each in order and uses the
# first that works. Verify current names at:
#   OpenRouter: https://openrouter.ai/models  (filter: image input)
#   Together:   https://api.together.ai/models
_PROVIDER_PRESETS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        # LIVE-TESTED on July 19, 2026 with find_vision_model.py /
        # preflight_check.py — both correctly rejected a car image
        # ("NO"). If these ever retire, run find_vision_model.py
        # again to auto-discover the current free vision models.
        "models": [
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "models": [
            "meta-llama/Llama-Vision-Free",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        ],
    },
}

_API_ENABLED = (
    bool(VISION_API_KEY)
    and "YAHAN" not in VISION_API_KEY
    and VISION_API_PROVIDER in _PROVIDER_PRESETS
)

_VISION_PROMPT = (
    "You are a STRICT triage filter for a dermatology screening tool. "
    "Look at the image and answer with EXACTLY ONE WORD from these "
    "three options:\n"
    "  DISEASE — ONLY if the image is a close-up of human skin AND a "
    "skin problem is CLEARLY visible: a distinct rash, mole, lesion, "
    "wound, infection, ulcer, patch of discoloration, or unusual "
    "growth.\n"
    "  NORMAL — if the image shows human skin that looks healthy or "
    "ordinary, with nothing clearly wrong. A plain hand, palm, "
    "fingers, arm, leg, or any body part with only hair, veins, "
    "creases, small freckles, or tiny ordinary marks MUST be NORMAL.\n"
    "  INVALID — if the image is not a skin close-up at all: a face, "
    "selfie, portrait, full-body photo, animal, vehicle, room, "
    "furniture, building, landscape, document, or object.\n"
    "If you are NOT SURE whether a real skin condition is present, "
    "answer NORMAL. Do not explain. Answer with one word only."
)

# Message shown when Pass C finds the skin looks healthy — clearer for
# the user than the generic invalid-image message.
HEALTHY_SKIN_MSG = (
    "No visible skin condition could be detected in this photo. "
    "Please upload a clear, close-up photo of the affected area only "
    "(the specific spot, rash, or lesion you are concerned about)."
)

# ── LOCAL Pass B — OpenCV Haar-cascade face detection ────────────────
# Ships inside opencv-python itself (no download, works offline).
if _CV2_OK:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _PROFILE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
else:
    _FACE_CASCADE = None
    _PROFILE_CASCADE = None


def _contains_face(image_path):
    """
    LOCAL face check — returns True if a human face (frontal or profile,
    either direction) is detected. A visible face means the photo is a
    selfie/portrait rather than a lesion close-up, so it is rejected.
    Instant, offline, zero tokens. Requires opencv-python.
    """
    if not _CV2_OK or _FACE_CASCADE is None or _FACE_CASCADE.empty():
        print("[streamlit_app.py] WARNING: opencv-python not installed — "
              "face check (Pass B) skipped. Run: pip install opencv-python")
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Downscale big photos so detection is fast and consistent.
    h, w = gray.shape
    if max(h, w) > 700:
        scale = 700.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    kwargs = dict(scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(_FACE_CASCADE.detectMultiScale(gray, **kwargs)) > 0:
        return True
    if _PROFILE_CASCADE is not None and not _PROFILE_CASCADE.empty():
        if len(_PROFILE_CASCADE.detectMultiScale(gray, **kwargs)) > 0:
            return True
        # Profile cascade only finds left-facing faces — mirror for right.
        flipped = cv2.flip(gray, 1)
        if len(_PROFILE_CASCADE.detectMultiScale(flipped, **kwargs)) > 0:
            return True
    return False


def _skin_pixel_ratio(image_path):
    """
    LOCAL Pass A — zero-cost, instant, no API, no tokens, works offline.

    Fraction of pixels that fall inside the classic human-skin color
    range in YCrCb color space. Cars, rooms, landscapes, documents and
    most animals score near 0; genuine skin close-ups score high.
    This lets the app reject obviously-irrelevant images IMMEDIATELY,
    without spending any API quota or waiting on any server.

    (It cannot reject faces — faces are skin-colored — which is exactly
    why Pass B below still exists for the images that get past this.)
    """
    img = Image.open(image_path).convert("RGB").resize((150, 150))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    mask = (cr > 133) & (cr < 180) & (cb > 77) & (cb < 127)
    return float(mask.mean())


# Below this ratio the image is rejected locally without calling the API.
# 0.30 = at least 30% of pixels must look like human skin. Lesion
# close-ups typically score 0.6+; cars/animals/rooms score near 0.
SKIN_RATIO_THRESHOLD = 0.30


def _resize_for_vision(image_path, max_side=384):
    """
    Downscale the image before sending to the vision API (Pass C).

    Vision APIs bill images by size (in tokens) — sending a full
    multi-MB camera photo burns hundreds of tokens per check and is
    slower to upload. A 384px thumbnail is more than enough for a
    YES/NO "is this skin?" decision, cuts token usage massively, and
    makes the API round-trip noticeably faster.
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def is_relevant_skin_image(image_path):
    """
    Image-content check — does this image actually show a close-up of
    human skin / a lesion, as opposed to a face, room, object, animal,
    vehicle, landscape, document, etc.?

    Runs AFTER is_skin_image() (which only rejects blank/dark frames)
    and BEFORE full_pipeline() / predict_disease() are ever called —
    so a non-skin image never reaches the CV model.

    v4 — FULLY LOCAL, three passes (see section comment for history):
      Pass A: skin-color pixel ratio (LOCAL) — instantly rejects cars,
              animals, rooms, landscapes, documents. 0 tokens, offline.
      Pass B: Haar-cascade face detection (LOCAL) — rejects selfies
              and portraits. 0 tokens, offline.
      Pass C: API semantic check (OPTIONAL) — OpenRouter or Together
              AI, whichever is configured above. Runs only if a real
              VISION_API_KEY was set. If the API errors here, the
              image is ALLOWED anyway, because it already passed both
              local checks — Pass C is a bonus, not a dependency. If
              no key is set, Pass C is skipped entirely.

    Every rejection path is deterministic and local, so this filter can
    no longer be silently disabled by a provider retiring a model,
    rate-limiting, or exhausting a quota — the failure mode that caused
    the original two-week silent bypass.

    Returns: (relevant: bool, reason: str)
        reason is INVALID_IMAGE_MSG when rejected, "" when allowed.
    """
    # ── Pass A: LOCAL skin-color check (instant, free, offline) ──────
    try:
        ratio = _skin_pixel_ratio(image_path)
        if ratio < SKIN_RATIO_THRESHOLD:
            print(
                f"[streamlit_app.py] Pass A rejected image "
                f"(skin-pixel ratio {ratio:.2f} < {SKIN_RATIO_THRESHOLD})"
            )
            return False, INVALID_IMAGE_MSG
    except Exception as e:
        # Local check failing is unusual; log and continue to Pass B.
        print(f"[streamlit_app.py] Pass A error (continuing): {e}")

    # ── Pass B: LOCAL face detection (instant, free, offline) ────────
    try:
        if _contains_face(image_path):
            print("[streamlit_app.py] Pass B rejected image (face detected "
                  "— portrait/selfie, not a lesion close-up)")
            return False, INVALID_IMAGE_MSG
    except Exception as e:
        print(f"[streamlit_app.py] Pass B error (continuing): {e}")

    # ── Pass C: API semantic check (OpenRouter / Together AI) ────────
    # Local passes confirm "this looks like skin with no face" — but
    # only a vision model can judge "is a visible CONDITION present?"
    # (e.g. reject healthy-skin close-ups). Runs ONLY if a key is set.
    if _API_ENABLED:
        preset = _PROVIDER_PRESETS[VISION_API_PROVIDER]
        try:
            img_bytes = _resize_for_vision(image_path)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            last_err = None
            for model_name in preset["models"]:
                try:
                    r = requests.post(
                        preset["base_url"] + "/chat/completions",
                        headers={
                            "Authorization": f"Bearer {VISION_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "max_tokens": 10,
                            "temperature": 0.0,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": _VISION_PROMPT},
                                    {"type": "image_url",
                                     "image_url": {
                                         "url": f"data:image/jpeg;base64,{img_b64}"
                                     }},
                                ],
                            }],
                        },
                        timeout=45,
                    )
                    if r.status_code != 200:
                        last_err = f"HTTP {r.status_code}: {r.text[:100]}"
                        print(f"[streamlit_app.py] Pass C model "
                              f"'{model_name}' failed: {last_err}")
                        continue
                    answer = (
                        r.json()["choices"][0]["message"]["content"] or ""
                    ).strip().upper()
                    # DEFAULT-DENY logic: ONLY the exact token "DISEASE"
                    # allows the image. NORMAL, INVALID, empty, or any
                    # unexpected reply all reject. (The old "reject only
                    # on NO, allow everything else" logic let a healthy
                    # hand through when the model answered loosely.)
                    if "DISEASE" in answer:
                        print(f"[streamlit_app.py] Pass C ({model_name}) "
                              f"APPROVED image (answer: {answer[:20]})")
                        return True, ""  # visible skin condition confirmed
                    if "NORMAL" in answer:
                        print(f"[streamlit_app.py] Pass C ({model_name}) "
                              f"rejected image (healthy skin, answer: "
                              f"{answer[:20]})")
                        return False, HEALTHY_SKIN_MSG
                    print(f"[streamlit_app.py] Pass C ({model_name}) "
                          f"rejected image (answer: {answer[:20]})")
                    return False, INVALID_IMAGE_MSG
                except Exception as e:
                    last_err = str(e)
                    print(f"[streamlit_app.py] Pass C model "
                          f"'{model_name}' error: {last_err[:100]}")
                    continue
            # All API models failed — FAIL CLOSED. Pass C is the only
            # layer that can distinguish healthy skin from a lesion, so
            # if it is unavailable the image cannot be safely verified.
            print(f"[streamlit_app.py] Pass C unavailable — blocking "
                  f"(fail closed): {str(last_err)[:100]}")
            return False, (
                "Image verification is temporarily unavailable (the "
                "vision service is busy), so this photo could not be "
                "checked. Please wait a minute and try again. "
                f"(Admin: Pass C error — {str(last_err)[:80]})"
            )
        except Exception as e:
            print(f"[streamlit_app.py] Pass C error — blocking "
                  f"(fail closed): {str(e)[:100]}")
            return False, (
                "Image verification is temporarily unavailable. "
                "Please wait a minute and try again. "
                f"(Admin: Pass C error — {str(e)[:80]})"
            )

    # Pass C disabled (no key set) — allow on the two local checks.
    return True, ""


# ─────────────────────────────────────────────
# MODEL LOADING — same initialize() call as app.py,
# wrapped in st.cache_resource so it only runs ONCE per server process
# instead of on every Streamlit script rerun. This is a Streamlit-runtime
# concern only; initialize()/pipeline.py internals are untouched.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_cv_model():
    print("\nStarting Skin Disease Chatbot (Streamlit)...")
    ok = initialize(MODEL_PATH)
    if not ok:
        print("WARNING: CV model not loaded.")
    return ok


st.set_page_config(
    page_title="Skin Disease Chatbot",
    page_icon="🔎",
    layout="wide",
)

model_loaded = load_cv_model()

# ─────────────────────────────────────────────
# LIGHT VISUAL POLISH (CSS only — no logic here)
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .result-card {
        border: 1px solid rgba(49, 130, 86, 0.25);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(49, 130, 86, 0.06);
        margin-bottom: 0.75rem;
    }
    .result-card.uncertain {
        border-color: rgba(200, 120, 20, 0.35);
        background: rgba(200, 120, 20, 0.07);
    }
    .result-name { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
    .result-conf { font-size: 0.85rem; opacity: 0.75; margin-bottom: 8px; }
    .uncertain-badge {
        display: inline-block; font-size: 0.75rem; font-weight: 600;
        color: #b3650f; background: rgba(200,120,20,0.12);
        padding: 2px 8px; border-radius: 999px; margin-bottom: 6px;
    }
    .disclaimer-text { font-size: 0.78rem; opacity: 0.65; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION STATE
# This is the direct replacement for Flask's `session` cookie.
# pipeline_session holds EXACTLY the same dict shape that new_session()
# / full_pipeline() / chat_followup() already expect and return.
# ─────────────────────────────────────────────

if "pipeline_session" not in st.session_state:
    st.session_state.pipeline_session = new_session()

if "chat_log" not in st.session_state:
    # Visual chat history only — separate from pipeline_session["history"],
    # exactly like the Flask version (the JS chat-window bubbles were also
    # independent of the server-side Groq history list).
    st.session_state.chat_log = []  # list of {"role": "user"|"bot", "text": str}

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Camera Capture / Retake support only. Purely a UI counter used to give
# st.camera_input() a fresh `key` on retake, which is the standard
# Streamlit way to discard a widget's current value and show an empty
# camera view again. Nothing else reads or depends on this value.
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0


def reset_everything():
    """Equivalent to POST /reset, plus clearing the visible chat —
    this is the new 'Clear Chat' button the requirements asked for.
    The Flask version had no UI button for this (only an internal call
    before each /analyze), so this also gives an explicit way to do it."""
    st.session_state.pipeline_session = new_session()
    st.session_state.chat_log = []
    st.session_state.analysis_result = None
    st.session_state.camera_key += 1  # also discard any in-progress camera capture


def format_bot_message(text):
    """
    Markdown-friendly version of the JS formatBotMessage() function in
    index.html. The 'Detected condition / Confidence / Urgency level /
    Medical note' block is now wrapped in a rounded card (matching the
    title bar and gauge card styling for a consistent professional
    look); the trailing disclaimer is italicized.
    Purely cosmetic — does not alter the text content itself.
    """
    if not text:
        return text

    disclaimer_marker = "IMPORTANT:"
    main_part, disclaimer_part = text, ""
    idx = text.find(disclaimer_marker)
    if idx != -1:
        main_part = text[:idx].strip()
        disclaimer_part = text[idx:].strip()

    detection_keys = ["Detected condition", "Confidence", "Urgency level", "Medical note"]
    box_lines = []
    out_lines = []
    for line in main_part.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        matched_key = next(
            (k for k in detection_keys if stripped.startswith(k + " :") or stripped.startswith(k + ":")),
            None,
        )
        if matched_key:
            colon_idx = stripped.find(":")
            key, val = stripped[: colon_idx + 1], stripped[colon_idx + 1 :]
            # Inside a raw-HTML card, markdown ** doesn't render —
            # use <strong> tags instead.
            box_lines.append(
                f'<div style="margin:3px 0;"><strong>{key}</strong>{val}</div>'
            )
        else:
            out_lines.append(stripped + "  ")

    formatted = ""
    if box_lines:
        formatted += (
            '<div style="border:1px solid rgba(255,255,255,0.12);'
            'border-radius:12px;background:rgba(255,255,255,0.03);'
            'padding:12px 16px;margin-bottom:10px;">'
            + "".join(box_lines)
            + "</div>\n\n"
        )
    formatted += "\n".join(out_lines)
    if disclaimer_part:
        formatted += f'\n\n<span class="disclaimer-text">{disclaimer_part}</span>'
    return formatted


# ─────────────────────────────────────────────
# SIDEBAR — image upload, symptoms, analyze, result card, quick questions
# (Direct equivalent of the Flask UI's left sidebar panel)
# ─────────────────────────────────────────────

QUICK_QUESTIONS = [
    "What are home remedies for this condition?",
    "Is this condition contagious?",
    "Should I see a doctor urgently?",
    "What should I avoid doing?",
    "Can this condition spread to other areas?",
    "What sunscreen is recommended?",
]

pending_message = None  # set below if a quick-question button is clicked

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:0.25rem;">'
        '<svg width="30" height="30" viewBox="0 0 48 48" style="flex-shrink:0;" aria-hidden="true">'
        '<circle cx="20" cy="20" r="13" fill="#E1F5EE" stroke="#1D9E75" stroke-width="4"/>'
        '<circle cx="23" cy="23" r="3.5" fill="#712B13"/>'
        '<circle cx="16" cy="15" r="1.8" fill="#993C1D"/>'
        '<line x1="30" y1="30" x2="42" y2="42" stroke="#1D9E75" stroke-width="5" stroke-linecap="round"/></svg>'
        '<span style="font-size:1.15rem;font-weight:700;">Skin Disease Assistant</span></div>',
        unsafe_allow_html=True,
    )
    if not model_loaded:
        st.warning("CV model not loaded — check MODEL_PATH in streamlit_app.py / pipeline.py.")

    # Pass C status — visible at all times so a missing/placeholder key
    # can never silently disable semantic verification again.
    if _API_ENABLED:
        st.caption(f"✅ Image verification (Pass C): ON — {VISION_API_PROVIDER}")
    else:
        st.error(
            "⚠ Image verification (Pass C) is OFF — VISION_API_KEY is "
            "not set in streamlit_app.py. Healthy-skin images will NOT "
            "be screened. Open streamlit_app.py, paste your key in "
            "VISION_API_KEY, save, and restart the app."
        )

    st.markdown("#### Skin Image")

    # ── Image source toggle — Upload Image / Capture Photo ─────────────
    # This is the ONLY structural addition to the input area. Both modes
    # ultimately produce the same UploadedFile-shaped object, which is
    # resolved below into a single `active_image` variable that the
    # Analyze handler consumes — the handler's own logic is unchanged.
    image_mode = st.radio(
        "Image source",
        ["📁 Upload Image", "📷 Capture Photo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None
    camera_file = None

    if image_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a clear, close-up photo of the affected area",
            type=sorted(ALLOWED_EXTENSIONS),
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Preview", use_container_width=True)

    else:
        # st.camera_input() requests camera permission via the browser's
        # native getUserMedia prompt (handled entirely by the browser/
        # Streamlit — no custom permission code needed). It works on
        # localhost (secure-context exemption) and on Streamlit Community
        # Cloud (served over HTTPS automatically), and shows its own
        # live preview + capture button on both desktop and mobile.
        camera_file = st.camera_input(
            "Take a clear, close-up photo of the affected area",
            key=f"camera_{st.session_state.camera_key}",
        )

        if camera_file is not None:
            # camera_input() already shows the captured snapshot itself —
            # this caption plus the Retake button give an explicit,
            # clearly-labeled way to discard it and capture again,
            # in addition to the widget's own small clear ("x") control.
            st.caption("Preview — captured photo shown above")
            if st.button("🔄 Retake Photo", use_container_width=True):
                st.session_state.camera_key += 1
                st.rerun()

    # Single unified image source for everything below — the Analyze
    # handler does not need to know whether this came from the uploader
    # or the camera; it just needs an UploadedFile-shaped object.
    active_image = uploaded_file if image_mode == "📁 Upload Image" else camera_file

    st.markdown("#### Describe Symptoms")
    symptoms_input = st.text_area(
        "Symptoms",
        placeholder="Describe what you see or feel... e.g. dark spot on arm, itchy for 2 weeks",
        height=90,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("✦ Analyze Skin", type="primary", use_container_width=True)

    # ── Result card — equivalent of #result-section / showResult() in JS ──
    result = st.session_state.analysis_result
    if result and result.get("success") and (result.get("disease_name") or result.get("uncertain")):
        st.markdown("#### Analysis Result")
        card_class = "result-card uncertain" if result.get("uncertain") else "result-card"
        if result.get("uncertain"):
            badge = '<div class="uncertain-badge">⚠ Inconclusive</div>'
            name = "Image unclear"
            conf_label = f"Confidence: {result.get('confidence')}% — below threshold"
        else:
            badge = ""
            name = result.get("disease_name", "—")
            conf_label = f"Confidence: {result.get('confidence')}%"

        st.markdown(
            f'<div class="{card_class}">{badge}'
            f'<div class="result-name">{name}</div>'
            f'<div class="result-conf">{conf_label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        # NOTE: The "All Probabilities" progress bars were moved from
        # here to the MAIN panel as a clinical urgency-colored bar
        # chart (see render_probability_chart below) — sidebar now
        # stays compact with just the headline result card.

    st.markdown("#### Quick Questions")
    qcols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        if qcols[i % 2].button(q, key=f"quick_{i}", use_container_width=True):
            pending_message = q

    st.divider()
    st.button("🗑 Clear Chat", on_click=reset_everything, use_container_width=True)

# ─────────────────────────────────────────────
# UI ADDITIONS (frontend-only) — derma-scan logo + clinical bar chart
# Chosen by the user from mockup options: Logo 1 (magnifier over a
# skin mole) and Graph A (urgency-colored clinical bars, moved from
# the sidebar into the main panel).
# ─────────────────────────────────────────────

def logo_svg(size=40):
    """Derma-scan magnifier logo (inline SVG — no image files needed)."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" '
        f'style="flex-shrink:0;" aria-hidden="true">'
        f'<circle cx="20" cy="20" r="13" fill="#E1F5EE" '
        f'stroke="#1D9E75" stroke-width="4"/>'
        f'<circle cx="23" cy="23" r="3.5" fill="#712B13"/>'
        f'<circle cx="16" cy="15" r="1.8" fill="#993C1D"/>'
        f'<line x1="30" y1="30" x2="42" y2="42" stroke="#1D9E75" '
        f'stroke-width="5" stroke-linecap="round"/></svg>'
    )


# Urgency level per disease full name (mirrors DISEASE_INFO in
# chatbot.py — duplicated here as display metadata only, so no
# backend file needs importing/altering for a UI concern).
URGENCY_BY_NAME = {
    "Melanoma":             "HIGH",
    "Basal Cell Carcinoma": "HIGH",
    "Actinic Keratoses":    "MEDIUM",
    "Benign Keratosis":     "LOW",
    "Melanocytic Nevi":     "LOW",
    "Dermatofibroma":       "LOW",
    "Vascular Lesions":     "LOW",
    "Vascular Lesion":      "LOW",
}

URGENCY_COLORS = {"HIGH": "#e24b4a", "MEDIUM": "#ef9f27", "LOW": "#639922"}


def gauge_card_html(result):
    """
    Graph C — gauge dashboard card, rendered WITH EACH analysis result
    inside the chat (so every prediction keeps its own graph, instead
    of a single top gauge that only reflected the latest analysis).
    Semicircular confidence gauge colored by urgency (red = HIGH,
    amber = MEDIUM, green = LOW) + urgency badge + condition name +
    the runner-up probability. Returns an HTML string.
    """
    name = result.get("disease_name") or "—"
    conf = float(result.get("confidence") or 0.0)
    urgency = URGENCY_BY_NAME.get(name, "LOW")
    color = URGENCY_COLORS[urgency]

    p = max(0.0, min(conf / 100.0, 1.0))
    theta = math.pi * (1.0 - p)
    x = 60 + 50 * math.cos(theta)
    y = 65 - 50 * math.sin(theta)
    arc = ""
    if p >= 0.01:
        arc = (
            f'<path d="M 10 65 A 50 50 0 0 1 {x:.1f} {y:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="12" stroke-linecap="round"/>'
        )

    probs = result.get("all_probs") or {}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    second = ""
    if len(ranked) > 1:
        second = f"Next: {ranked[1][0]} — {float(ranked[1][1]):.1f}%"

    return (
        f'<div style="border:1px solid rgba(255,255,255,0.12);'
        f'border-radius:12px;background:rgba(255,255,255,0.03);'
        f'padding:12px 16px;display:flex;'
        f'align-items:center;gap:14px;">'
        f'<svg viewBox="0 0 120 72" width="118" style="flex-shrink:0;" '
        f'aria-hidden="true">'
        f'<path d="M 10 65 A 50 50 0 0 1 110 65" fill="none" '
        f'stroke="rgba(128,128,128,0.28)" stroke-width="12" '
        f'stroke-linecap="round"/>'
        f'{arc}'
        f'<text x="60" y="58" text-anchor="middle" '
        f'style="font-size:15px;font-weight:700;fill:currentColor;">'
        f'{conf:.1f}%</text></svg>'
        f'<div>'
        f'<div style="display:inline-block;font-size:0.72rem;'
        f'font-weight:600;color:{color};background:{color}26;'
        f'padding:2px 10px;border-radius:999px;margin-bottom:5px;">'
        f'Urgency: {urgency}</div>'
        f'<div style="font-weight:700;font-size:0.95rem;">{name}</div>'
        + (
            f'<div style="font-size:0.76rem;opacity:0.7;margin-top:2px;">'
            f'{second}</div>' if second else ''
        )
        + '</div></div>'
    )


# ─────────────────────────────────────────────
# MAIN PANEL — chat window
# ─────────────────────────────────────────────

# Title bar — rounded professional card (Logo 1 + app name). The
# per-result gauge now lives inside each chat message instead of a
# single top-right slot, so every prediction keeps its own graph.
st.markdown(
    f'<div style="border:1px solid rgba(255,255,255,0.12);'
    f'border-radius:12px;background:rgba(255,255,255,0.03);'
    f'padding:14px 20px;display:flex;align-items:center;gap:14px;'
    f'margin-bottom:0.5rem;">'
    f'{logo_svg(46)}'
    f'<h1 style="margin:0;font-size:2.0rem;">Skin Disease Chatbot Assistant</h1>'
    f'</div>',
    unsafe_allow_html=True,
)

if not st.session_state.chat_log:
    st.info(
        "Upload a photo or capture one with your camera, then describe your symptoms. "
        "I'll help identify the condition and guide you on next steps.\n\n"
        "I am a screening tool only — always consult a qualified dermatologist "
        "for proper diagnosis and treatment."
    )

for msg in st.session_state.chat_log:
    avatar = "🔎" if msg["role"] == "bot" else "🧑"
    with st.chat_message("assistant" if msg["role"] == "bot" else "user", avatar=avatar):
        if msg["role"] == "bot":
            _mres = msg.get("result")
            if _mres:
                # Result message: detection text on the left, this
                # result's OWN gauge card on the right — the gauge
                # scrolls with its message and never goes stale.
                _mc1, _mc2 = st.columns([2, 1], vertical_alignment="top")
                with _mc1:
                    st.markdown(format_bot_message(msg["text"]), unsafe_allow_html=True)
                with _mc2:
                    st.markdown(gauge_card_html(_mres), unsafe_allow_html=True)
            else:
                st.markdown(format_bot_message(msg["text"]), unsafe_allow_html=True)
        else:
            st.markdown(msg["text"])

# (Result visualization: each analysis message now carries its own
# Graph C gauge card — see gauge_card_html and the chat render loop.)

# ─────────────────────────────────────────────
# ANALYZE HANDLER
# Equivalent of: await fetch('/reset'); await fetch('/analyze', {...})
#
# Reads `active_image` (resolved above from either the uploader or the
# camera) instead of `uploaded_file` directly. A NEW Check 2 — the
# is_relevant_skin_image() semantic guardrail — runs after the existing
# Check 1 (is_skin_image()) and before full_pipeline() is ever called.
# Everything from full_pipeline() onward, including the call into
# prediction / translation / chatbot logic, is completely unchanged.
# ─────────────────────────────────────────────

if analyze_clicked:
    if active_image is None:
        st.sidebar.error("Please upload or capture a skin image first.")
    elif image_mode == "📁 Upload Image" and not allowed_file(active_image.name):
        st.sidebar.error("Invalid file type. Use JPG, PNG, or WEBP.")
    else:
        with st.spinner("Analyzing..."):
            # Save temporarily — same pattern as app.py (unique filename, deleted after use).
            # Camera captures are always JPEG, so a fixed filename is used for that source
            # instead of relying on active_image.name (which Streamlit's camera_input does
            # not guarantee to expose the same way file_uploader does across versions) —
            # this keeps behavior identical and predictable in both environments.
            source_name = active_image.name if image_mode == "📁 Upload Image" else "camera_capture.jpg"
            unique_fn = f"{uuid.uuid4().hex}_{source_name}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_fn)
            with open(image_path, "wb") as f:
                f.write(active_image.getbuffer())

            try:
                # Check 1: not blank or pitch black — identical to app.py /analyze
                valid, reason = is_skin_image(image_path)
                if not valid:
                    st.session_state.chat_log.append({"role": "bot", "text": "⚠ " + reason})
                else:
                    # Check 2 (NEW): is this actually a skin/lesion close-up, as
                    # opposed to a face, room, object, vehicle, etc.? This is the
                    # semantic filter identified as missing in the image-guardrail
                    # investigation. Runs only after Check 1 passes, and only
                    # before full_pipeline() / predict_disease() are ever called.
                    relevant, relevance_reason = is_relevant_skin_image(image_path)
                    if not relevant:
                        st.session_state.chat_log.append({"role": "bot", "text": "⚠ " + relevance_reason})
                    else:
                        # Every Analyze click starts a fresh pipeline session —
                        # this mirrors app.py exactly: it calls /reset, AND then
                        # creates a brand-new session inside /analyze itself.
                        ps = new_session()
                        result, ps = full_pipeline(
                            image_input=image_path,
                            symptom_text=symptoms_input,
                            session=ps,
                        )
                        st.session_state.pipeline_session = ps

                        if not result.get("success"):
                            st.session_state.chat_log.append(
                                {"role": "bot", "text": "⚠ " + (result.get("error") or "Analysis failed. Please try again.")}
                            )
                        else:
                            st.session_state.analysis_result = result
                            # Attach this result's data to its own chat
                            # message so its gauge renders WITH it and
                            # stays correct even after later analyses.
                            _snap = None
                            if result.get("disease_name") and not result.get("uncertain"):
                                _snap = {
                                    "disease_name": result.get("disease_name"),
                                    "confidence":   result.get("confidence"),
                                    "all_probs":    result.get("all_probs"),
                                }
                            st.session_state.chat_log.append({
                                "role": "bot",
                                "text": result.get("chatbot_response", ""),
                                "result": _snap,
                            })

            except Exception as e:
                st.session_state.chat_log.append({"role": "bot", "text": f"⚠ Analysis failed: {e}"})
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        st.rerun()

# ─────────────────────────────────────────────
# CHAT HANDLER
# Equivalent of: await fetch('/chat', {...})
# Shared by both the chat input box and the quick-question buttons,
# exactly like sendChat()/quickAsk() shared logic in index.html.
# ─────────────────────────────────────────────

def handle_user_message(message):
    st.session_state.chat_log.append({"role": "user", "text": message})
    with st.spinner("Thinking..."):
        # chat_followup() already returns the "upload an image first" message
        # internally when disease_label is None — no need to duplicate that
        # check here, it's handled identically to the Flask /chat route.
        response, ps = chat_followup(message, st.session_state.pipeline_session)
        st.session_state.pipeline_session = ps
    st.session_state.chat_log.append({"role": "bot", "text": response})


chat_box_message = st.chat_input("Ask a follow-up question about your skin condition...")

if pending_message:
    handle_user_message(pending_message)
    st.rerun()
elif chat_box_message:
    handle_user_message(chat_box_message)
    st.rerun()
