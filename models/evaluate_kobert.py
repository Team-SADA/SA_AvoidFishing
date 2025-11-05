# file: models/evaluate_kobert.py
# ============================================
# KoBERT 평가 스크립트
# ============================================

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 설정
# -----------------------------
MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 96
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 경로
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "KorCCViD_v1.3_fullcleansed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

# -----------------------------
# 데이터셋 클래스
# -----------------------------
class KoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -----------------------------
# 평가 함수
# -----------------------------
def evaluate_kobert():
    print("KoBERT 모델 평가 시작")

    # 최신 모델 로드
    model_files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith("kobert") and f.endswith(".pt")],
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("학습된 KoBERT 모델(.pt)을 찾을 수 없습니다.")
    model_path = os.path.join(MODEL_DIR, model_files[0])
    print(f" 최신 모델: {model_path}")

    # 데이터 로드
    df = pd.read_csv(DATA_PATH)
    if "Transcript" in df.columns:
        df.rename(columns={'Transcript': 'text', 'Label': 'label'}, inplace=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = KoBERTDataset(X_test, y_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    report = classification_report(labels, preds, digits=4)

    print("\n KoBERT 분류 리포트:\n", report)
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

if __name__ == "__main__":
    evaluate_kobert()
