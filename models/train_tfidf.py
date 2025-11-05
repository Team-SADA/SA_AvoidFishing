# file: models/train_tfidf_v3.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------------------------------
#  경로 설정
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
DANGER_PATH = os.path.join(BASE_DIR, "data", "processed", "danger_words.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizer")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# -------------------------------------------------------
#  데이터 로드
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].astype(str)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Train: {len(X_train)} / Test: {len(X_test)}")

# -------------------------------------------------------
#  위험 단어 리스트 불러오기
# -------------------------------------------------------
if os.path.exists(DANGER_PATH):
    with open(DANGER_PATH, "r", encoding="utf-8") as f:
        danger_words = [line.strip() for line in f.readlines()]
else:
    danger_words = []

print(f" 위험 단어 {len(danger_words)}개 로드 완료")

# -------------------------------------------------------
# 4️⃣ TF-IDF 벡터화
# -------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    stop_words=["을", "를", "이", "가", "은", "는", "에서", "으로", "에게"]
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f" TF-IDF vectorized: {X_train_tfidf.shape[1]} features")

# -------------------------------------------------------
#  모델 구성 (Voting 앙상블)
# ------------------------------------------------------
log_reg = LogisticRegression(max_iter=300, class_weight="balanced")
rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")

model = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf)],
    voting='soft'  # 확률 기반 앙상블
)
model.fit(X_train_tfidf, y_train)
print(" VotingClassifier 학습 완료")

# -------------------------------------------------------
# 예측 + 위험 단어 가중치 적용
# -------------------------------------------------------
base_probs = model.predict_proba(X_test_tfidf)[:, 1]

def danger_word_boost(text, base_prob):
    danger_count = sum(word in text for word in danger_words)
    if danger_count > 0:
        # 위험 단어가 많을수록 가중치 증가
        boosted_prob = min(base_prob * (1.2 + 0.06 * danger_count), 1.0)
        return boosted_prob
    return base_prob

adjusted_probs = np.array([
    danger_word_boost(t, p) for t, p in zip(X_test, base_probs)
])

y_pred = (adjusted_probs >= 0.5).astype(int)

# -------------------------------------------------------
#  평가
# -------------------------------------------------------
print("\n 분류 리포트:\n", classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"최종 정확도: {acc:.4f}")

# -------------------------------------------------------
#저장
# -------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(model, os.path.join(MODEL_DIR, f"tfidf_hybrid_{timestamp}.pkl"))
joblib.dump(vectorizer, os.path.join(TOKENIZER_DIR, f"tfidf_vectorizer_{timestamp}.pkl"))

print(f"\n 모델 저장 완료: {MODEL_DIR}")
print(f" 벡터 저장 완료: {TOKENIZER_DIR}")
print(" 학습 완료: TF-IDF + VotingClassifier + 위험단어가중 Hybrid")
