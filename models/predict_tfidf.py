# file: models/predict_tfidf_v3.py

import os
import joblib
import numpy as np

# -------------------------------------------------------
# 1️⃣ 경로 설정
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "outputs", "tokenizer")
DANGER_PATH = os.path.join(BASE_DIR, "data", "processed", "danger_words.txt")

# -------------------------------------------------------
# 2️⃣ 가장 최신 모델과 벡터 불러오기
# -------------------------------------------------------
def get_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"{directory} 안에 {prefix} 관련 파일이 없습니다.")
    latest = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest)

model_path = get_latest_file(MODEL_DIR, "tfidf_hybrid")
vectorizer_path = get_latest_file(TOKENIZER_DIR, "tfidf_vectorizer")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print(f"✅ 모델 로드 완료: {os.path.basename(model_path)}")
print(f"✅ 벡터 로드 완료: {os.path.basename(vectorizer_path)}")

# -------------------------------------------------------
# 3️⃣ 위험 단어 리스트 로드
# -------------------------------------------------------
if os.path.exists(DANGER_PATH):
    with open(DANGER_PATH, "r", encoding="utf-8") as f:
        danger_words = [line.strip() for line in f.readlines()]
else:
    danger_words = []
print(f"⚠️ 위험 단어 {len(danger_words)}개 로드 완료")

# -------------------------------------------------------
# 4️⃣ 예측 함수
# -------------------------------------------------------
def predict_text_tfidf(text: str):
    text_tfidf = vectorizer.transform([text])
    prob = model.predict_proba(text_tfidf)[0][1]  # 클래스 1 = 피싱

    # 위험 단어 포함 시 확률 보정
    danger_count = sum(word in text for word in danger_words)
    if danger_count > 0:
        prob = min(prob * (1.2 + 0.1 * danger_count), 1.0)

    label = "보이스피싱 의심" if prob >= 0.5 else "정상"

    print("\n💬 입력 문장:", text)
    print(f"🎯 예측 결과: {label} ({prob * 100:.2f}% 확률)")

    # 추가 판별 안내
    if prob >= 0.7:
        print("⚠️ 추가 판별 필요: KoBERT 모델로 정밀 검증을 권장합니다.")

# -------------------------------------------------------
# 5️⃣ 실행 루프
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n📞 문장을 입력하세요 (종료: exit):")
    while True:
        text = input("\n> ")
        if text.lower() == "exit":
            break
        predict_text_tfidf(text)
