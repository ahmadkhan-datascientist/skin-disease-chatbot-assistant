# Skin Disease Chatbot Assistant
Web-based AI chatbot for skin disease detection using Image + Symptoms fusion. User uploads skin image + types symptoms in Urdu/English -> CNN reads image + BERT reads symptoms -> Fusion model predicts disease + confidence + advice + triage.

## Tech
Python, PyTorch, ResNet, BERT, Streamlit, Urdu Translation

## My Contribution (NLP)
BERT symptom classification, Urdu-English translation, chatbot responses

## How to Run
1. pip install -r requirements.txt
2. Download model from https://drive.google.com/file/d/1rwWd5QEle9tiY1wjwo1I3BZxFt_T837f/view?usp=sharing and put skin_model_best.pth in project folder
3. streamlit run streamlit_app.py

## Disclaimer
Academic prototype only, not for clinical use.
