# file: main.py
# ============================================
# ⚡ Voice Phishing Detection Unified System
# TF-IDF + KoBERT 결합 (50% 이상 시 추가 판별)
# ============================================

import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------
# ⚙️ 설정
# -------------------------------------------------------
MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# 📂 경로 설정
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "outputs", "tokenizer")
DANGER_PATH = os.path.join(BASE_DIR, "data", "processed", "danger_words.txt")

# -------------------------------------------------------
# 🧩 최신 파일 로드 함수
# -------------------------------------------------------
def get_latest_file(directory, prefix, extension):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"{directory} 내에 {prefix} 관련 파일이 없습니다.")
    latest = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest)

# -------------------------------------------------------
# 🧠 TF-IDF 모델 로드
# -------------------------------------------------------
def load_tfidf_model():
    model_path = get_latest_file(MODEL_DIR, "tfidf", ".pkl")
    vectorizer_path = get_latest_file(TOKENIZER_DIR, "tfidf", ".pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print(f"✅ TF-IDF 모델 로드 완료: {os.path.basename(model_path)}")
    print(f"✅ TF-IDF 벡터 로드 완료: {os.path.basename(vectorizer_path)}")

    return model, vectorizer

# -------------------------------------------------------
# 🧠 KoBERT 모델 로드
# -------------------------------------------------------
def load_kobert_model():
    model_path = get_latest_file(MODEL_DIR, "kobert", ".pt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"✅ KoBERT 모델 로드 완료: {os.path.basename(model_path)}")
    return tokenizer, model

# -------------------------------------------------------
# ⚠️ 위험 단어 로드
# -------------------------------------------------------
def load_danger_words():
    if os.path.exists(DANGER_PATH):
        with open(DANGER_PATH, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f.readlines()]
        print(f"⚠️ 위험 단어 {len(words)}개 로드 완료")
        return words
    print("⚠️ 위험 단어 파일이 없습니다. 건너뜁니다.")
    return []

# -------------------------------------------------------
# 🔍 TF-IDF 예측
# -------------------------------------------------------
def predict_tfidf(text, model, vectorizer, danger_words):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]  # class 1 = phishing

    # 위험 단어 포함 시 확률 보정
    danger_count = sum(word in text for word in danger_words)
    if danger_count > 0:
        prob = min(prob * (1.2 + 0.1 * danger_count), 1.0)

    return prob * 100  # %

# -------------------------------------------------------
# 🔍 KoBERT 예측
# -------------------------------------------------------
def predict_kobert(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        phishing_prob = probs[1].item() * 100

    return phishing_prob

# -------------------------------------------------------
# ⚡ 통합 예측 로직
# -------------------------------------------------------
def unified_prediction(text, tfidf_model, vectorizer, tokenizer, kobert_model, danger_words):
    tfidf_prob = predict_tfidf(text, tfidf_model, vectorizer, danger_words)
    print(f"\n⚡ TF-IDF 예측 확률: {tfidf_prob:.2f}%")

    # TF-IDF 결과가 50% 이상이면 KoBERT 실행
    if tfidf_prob >= 50:
        kobert_prob = predict_kobert(text, tokenizer, kobert_model)
        print(f"🧠 KoBERT 예측 확률: {kobert_prob:.2f}%")
        final_prob = (tfidf_prob * 0.6) + (kobert_prob * 0.4)
    else:
        final_prob = tfidf_prob

    label = "⚠️ 보이스피싱 의심" if final_prob >= 50 else "✅ 정상 통화"

    print(f"\n💬 입력 문장: {text}")
    print(f"🎯 최종 판별 결과: {label} ({final_prob:.2f}% 확률)")

    if final_prob >= 70:
        print("🚨 추가 검증 권장: 고위험 문장으로 감지되었습니다.\n")

# -------------------------------------------------------
# 🧠 실행 루프
# -------------------------------------------------------
if __name__ == "__main__":
    print("🚀 하이브리드 보이스피싱 탐지 시스템 시작\n")

    # 모델 로드
    tfidf_model, vectorizer = load_tfidf_model()
    tokenizer, kobert_model = load_kobert_model()
    danger_words = load_danger_words()

    # 실시간 예측 루프
    while True:
        text = input("\n📞 문장을 입력하세요 (종료: exit): ").strip()
        if text.lower() == "exit":
            print("👋 시스템을 종료합니다.")
            break

        unified_prediction(text, tfidf_model, vectorizer, tokenizer, kobert_model, danger_words)
