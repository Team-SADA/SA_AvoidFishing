# file: models/evaluate_tfidf.py

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocessing.features import get_tfidf_features
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 경로 설정
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "outputs", "tokenizer")

# 최신 모델 파일 자동 탐색
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("tfidf_logistic")])
vectorizer_files = sorted([f for f in os.listdir(TOKENIZER_DIR) if f.startswith("tfidf_vectorizer")])

assert model_files and vectorizer_files, "모델 또는 벡터 파일을 찾을 수 없습니다."

model_path = os.path.join(MODEL_DIR, model_files[-1])
vectorizer_path = os.path.join(TOKENIZER_DIR, vectorizer_files[-1])

print(f"최신 모델: {model_path}")
print(f"최신 벡터: {vectorizer_path}")

# -------------------------------------------------------
# 데이터 로드 및 분할
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# -------------------------------------------------------
# 모델 및 벡터 불러오기
# -------------------------------------------------------
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -------------------------------------------------------
# 예측 및 평가
# -------------------------------------------------------
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

print("\n분류 리포트:\n", classification_report(y_test, y_pred))

# Confusion matrix 시각화
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["정상", "피싱"], yticklabels=["정상", "피싱"])
plt.title("TF-IDF Confusion Matrix")
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.show()
