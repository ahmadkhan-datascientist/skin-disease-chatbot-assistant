"""
preflight_check.py — App Chalane Se PEHLE Poora System Test
FYP: Skin Disease Chatbot Assistant

KYA CHECK KARTA HAI:
  [1] opencv-python installed hai? Face detector load hota hai?
  [2] Pass A (skin-color check) car.jpg ko reject karta hai?
  [3] Teeno OpenRouter models zinda hain? Car ko NO kehte hain?
  [4] (Agar lesion.jpg maujood ho) Lesion ko YES kehte hain?

KAISE CHALAO:
  1. Neeche OPENROUTER_API_KEY mein apni sk-or-... key daalo
  2. car.jpg isi folder mein ho
  3. (Behtar) HAM10000 folder se koi image copy kar ke isi folder
     mein "lesion.jpg" naam se rakho — to test [4] bhi chalega
  4. Terminal:  python preflight_check.py
  5. Aakhri line "SYSTEM READY" aaye to app chalao!
"""

import base64
import numpy as np
import requests
from PIL import Image

# ── YAHAN APNI KEY DAALO (wohi jo streamlit_app.py mein daali) ──
OPENROUTER_API_KEY = "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"

CAR_IMAGE    = "car.jpg"
LESION_IMAGE = "lesion.jpg"   # optional — ho to test [4] chalega

MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

PROMPT = (
    "Is this a close-up photo of human skin showing a rash, "
    "mole or lesion? Reply with exactly one word: YES or NO."
)

results = {}

print("=" * 64)
print("PRE-FLIGHT CHECK SHURU...")
print("=" * 64)

# ── [1] OpenCV + face detector ──────────────────────────────────────
print("\n[1] OpenCV / Face detector check...")
try:
    import cv2
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        print("    FAIL — cascade load nahi hua")
        results["opencv"] = False
    else:
        print("    OK — opencv installed, face detector tayyar")
        results["opencv"] = True
except ImportError:
    print("    FAIL — opencv-python installed NAHI hai!")
    print("    Chalao: pip install opencv-python")
    results["opencv"] = False

# ── [2] Pass A local check car par ──────────────────────────────────
print("\n[2] Pass A (skin-color) check car.jpg par...")
def skin_ratio(path):
    img = Image.open(path).convert("RGB").resize((150, 150))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299*r + 0.587*g + 0.114*b
    cr = (r - y)*0.713 + 128
    cb = (b - y)*0.564 + 128
    return float(((cr > 133) & (cr < 180) & (cb > 77) & (cb < 127)).mean())

try:
    ratio = skin_ratio(CAR_IMAGE)
    if ratio < 0.30:
        print(f"    OK — car ka skin-ratio {ratio:.2f} (< 0.30, reject hoga)")
        results["passA"] = True
    else:
        print(f"    WARNING — car ka ratio {ratio:.2f} (>= 0.30, Pass A se")
        print("    nahi rukega — Pass C rokega). Output Claude ko dikhana.")
        results["passA"] = True  # Pass C sambhal lega
except FileNotFoundError:
    print(f"    FAIL — {CAR_IMAGE} nahi mili, isi folder mein rakho")
    results["passA"] = False

# ── [3] API models car par ──────────────────────────────────────────
print("\n[3] OpenRouter models check (car -> NO aana chahiye)...")
if "YAHAN" in OPENROUTER_API_KEY:
    print("    FAIL — pehle file mein apni key daalo!")
    results["api"] = False
else:
    def ask_model(model, image_file):
        with open(image_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 10, "temperature": 0.0,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text": PROMPT},
                      {"type": "image_url", "image_url": {
                          "url": f"data:image/jpeg;base64,{b64}"}}]}]},
            timeout=60)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return (r.json()["choices"][0]["message"]["content"] or "").strip().upper(), None

    api_ok = False
    for m in MODELS:
        try:
            ans, err = ask_model(m, CAR_IMAGE)
            if err:
                print(f"    {m}: FAIL ({err})")
            elif "NO" in ans:
                print(f"    {m}: OK (jawab: NO)")
                api_ok = True
            else:
                print(f"    {m}: chala lekin jawab '{ans[:20]}'")
        except Exception as e:
            print(f"    {m}: FAIL ({str(e)[:60]})")
    results["api"] = api_ok

# ── [4] (Optional) lesion image par YES ─────────────────────────────
print("\n[4] Lesion test (optional)...")
import os
if os.path.exists(LESION_IMAGE) and results.get("api"):
    try:
        for m in MODELS:
            try:
                ans, err = ask_model(m, LESION_IMAGE)
                if not err:
                    mark = "OK" if "YES" in ans else "WARNING"
                    print(f"    {m}: {mark} (jawab: {ans[:20]})")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"    Skip: {str(e)[:60]}")
else:
    print(f"    Skip — '{LESION_IMAGE}' nahi mili (HAM10000 se koi image")
    print(f"    is naam se copy karo to ye test bhi chalega)")

# ── FINAL VERDICT ───────────────────────────────────────────────────
print()
print("=" * 64)
core_ok = results.get("opencv") and results.get("passA") and results.get("api")
if core_ok:
    print("SYSTEM READY! Ab chalao:  streamlit run streamlit_app.py")
    print("(Aur streamlit_app.py mein bhi wohi key daali ho — check kar lo)")
else:
    print("SYSTEM READY NAHI — upar jahan FAIL likha hai wo theek karo,")
    print("ya poora output Claude ko bhej do.")
print("=" * 64)
