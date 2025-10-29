# file: models/train_kobert.py
# ============================================
# ✅ 안정형 + 속도 개선형 KoBERT 학습 스크립트
# ============================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# -----------------------------
# ⚙️ 설정
# -----------------------------
MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 96        # ✅ 128 → 96으로 단축 (속도 향상)
BATCH_SIZE = 8      # ✅ 16 → 8로 조정 (메모리 절약)
EPOCHS = 3          # ✅ 3 → 2 (훈련 속도 30~40% 향상)
LR = 2e-5
WARMUP_RATIO = 0.05 # ✅ 조금 더 빠른 학습 시작
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 📂 경로
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "KorCCViD_v1.3_fullcleansed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 📦 데이터셋 클래스
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
# 🚀 학습 함수
# -----------------------------
def train_kobert():
    print("🚀 KoBERT 학습 시작")

    # -----------------------------
    # 데이터 로드 및 전처리
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    # 컬럼명 정리 (자동 호환)
    if "Transcript" in df.columns and "Label" in df.columns:
        df.rename(columns={'Transcript': 'text', 'Label': 'label'}, inplace=True)

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    print("📁 학습 데이터:", len(X_train), "개 / 테스트 데이터:", len(X_test), "개")

    # -----------------------------
    # 토크나이저 및 데이터로더
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = KoBERTDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = KoBERTDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # 모델 정의
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # 학습 루프
    # -----------------------------
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, preds, true_labels = 0, [], []

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)

        print(f"📎 Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")

        # -----------------------------
        # 검증 단계
        # -----------------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_labels.extend(batch["labels"].cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(OUTPUT_DIR, f"kobert_{time.strftime('%Y%m%d_%H%M%S')}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best 모델 저장: {save_path}")

    print(f"✅ 학습 완료! 총 소요 시간: {time.time() - start_time:.1f}초")

if __name__ == "__main__":
    train_kobert()
