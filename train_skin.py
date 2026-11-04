# -*- coding: utf-8 -*-
"""
train_skin.py — ResNet-50 on HAM10000
FYP: Skin Disease Chatbot Assistant (BS Data Science)

CHANGES FROM FRIEND'S COLAB VERSION:
  1. Removed all google.colab imports (drive.mount etc)
  2. Removed cv2 import (not used anywhere in the code)
  3. Changed all Colab paths to local D:\Fyp Project paths
  4. Added load_model() function — used by pipeline.py
  5. Added predict_disease() function — used by pipeline.py
  6. Model saved as skin_model_best.pth to match pipeline.py

HOW TO USE:
  - Training  → run on Colab (needs GPU), download skin_model_best.pth
  - Inference → pipeline.py imports load_model() and predict_disease()
"""

import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────

np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# ─────────────────────────────────────────────
# 2. PATHS — local laptop paths
# Change these if your folder is different
# ─────────────────────────────────────────────

BASE_DIR     = r"D:\Fyp Project"
IMG_DIR_1    = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2    = os.path.join(BASE_DIR, "HAM10000_images_part_2")
METADATA_CSV = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
SAVE_PATH    = os.path.join(BASE_DIR, "skin_model_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 3. DISEASE LABELS
# ─────────────────────────────────────────────

lesion_type_dict = {
    'nv':    'Melanocytic nevi',
    'mel':   'Melanoma',
    'bkl':   'Benign keratosis-like lesions',
    'bcc':   'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc':  'Vascular lesions',
    'df':    'Dermatofibroma',
}

# Maps class index → short label (used in predict_disease)
IDX_TO_LABEL = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc',
}

FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions',
}

# ─────────────────────────────────────────────
# 4. NORMALIZATION — HAM10000-specific values
# ─────────────────────────────────────────────

norm_mean = [0.485, 0.456, 0.406]
norm_std  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# 5. DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data():
    # Get all images from both folders
    image_paths_part1 = glob(os.path.join(IMG_DIR_1, "*.jpg"))
    image_paths_part2 = glob(os.path.join(IMG_DIR_2, "*.jpg"))
    all_image_paths   = image_paths_part1 + image_paths_part2

    print(f"Total images found: {len(all_image_paths)}")

    # Map image_id to full path
    imageid_path_dict = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in all_image_paths
    }

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    df['path'] = df['image_id'].map(imageid_path_dict.get)

    # Remove rows where image file not found
    df = df[df['path'].notnull()].reset_index(drop=True)

    # Add labels
    df['cell_type']     = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

    # Remove duplicate lesions
    df_undup = df.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1].reset_index()

    df['duplicates'] = df['lesion_id'].apply(
        lambda x: 'unduplicated' if x in df_undup['lesion_id'].values else 'duplicated'
    )
    df_undup = df[df['duplicates'] == 'unduplicated']

    print(f"Unique lesions: {len(df_undup)}")
    print("\nClass distribution:")
    print(df_undup['cell_type'].value_counts())

    # 3-way split: train / val / test
    y = df_undup['cell_type_idx']
    df_train_full, df_val = train_test_split(
        df_undup, test_size=0.2, stratify=y, random_state=42
    )
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)

    df_train_full = df_train_full.reset_index(drop=True)
    df_val        = df_val.reset_index(drop=True)
    df_test       = df_test.reset_index(drop=True)

    print(f"\nTrain: {len(df_train_full)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Data balancing — oversample minority classes
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    df_train = df_train_full.copy()
    for i in range(7):
        class_subset = df_train_full[df_train_full['cell_type_idx'] == i]
        if len(class_subset) > 0 and data_aug_rate[i] > 1:
            df_train = pd.concat(
                [df_train] + [class_subset] * (data_aug_rate[i] - 1),
                ignore_index=True
            )

    df_train = df_train.reset_index(drop=True)
    print(f"After augmentation — Train: {len(df_train)}")

    return df_train, df_val, df_test

# ─────────────────────────────────────────────
# 6. TRANSFORMS
# ─────────────────────────────────────────────

input_size = 224

train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# ─────────────────────────────────────────────
# 7. DATASET CLASS
# ─────────────────────────────────────────────

class HAM10000(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        image    = Image.open(img_path).convert('RGB')
        label    = torch.tensor(int(self.df.loc[idx, 'cell_type_idx']))
        if self.transform:
            image = self.transform(image)
        return image, label

# ─────────────────────────────────────────────
# 8. MODEL — ResNet-50
# ─────────────────────────────────────────────

def build_model():
    model    = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# 9. TRAINING
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("=" * 50)

    df_train, df_val, df_test = prepare_data()

    train_loader = DataLoader(
        HAM10000(df_train, train_transform),
        batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        HAM10000(df_val, val_transform),
        batch_size=32, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        HAM10000(df_test, val_transform),
        batch_size=32, shuffle=False, num_workers=2
    )

    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    best_val_acc = 0.0

    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs        = model(images)
                _, preds       = torch.max(outputs, 1)
                total         += labels.size(0)
                correct       += (preds == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model (val={val_acc*100:.2f}%)")

    # Final test evaluation
    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print("\nFinal evaluation on TEST SET:")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(
        all_labels, all_preds,
        target_names=list(lesion_type_dict.values())
    ))
    print("Model saved at:", SAVE_PATH)

# ─────────────────────────────────────────────
# 10. INFERENCE FUNCTIONS
# Used by pipeline.py — do not remove
# ─────────────────────────────────────────────

def load_model(model_path=SAVE_PATH):
    """
    Load trained ResNet-50 for inference.
    Called ONCE at Flask startup in pipeline.py.
    """
    model    = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def predict_disease(image_input, model):
    """
    Predict skin disease from an image.

    Args:
        image_input: file path (str), PIL Image, or file-like object
        model:       loaded model from load_model()

    Returns:
        dict:
            label      — short code: "mel", "nv" etc.
            full_name  — "Melanoma"
            confidence — float 0-100
            uncertain  — True if confidence < 60%
            all_probs  — dict of all 7 class probabilities
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100
    label      = IDX_TO_LABEL[top_idx]

    all_probs = {
        FULL_NAMES[IDX_TO_LABEL[i]]: round(float(probs[i]) * 100, 1)
        for i in range(7)
    }

    return {
        "label":      label,
        "full_name":  FULL_NAMES[label],
        "confidence": round(confidence, 1),
        "uncertain":  confidence < 60.0,
        "all_probs":  all_probs,
    }


if __name__ == "__main__":
    main()
