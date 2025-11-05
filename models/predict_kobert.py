# file: models/predict_kobert.py
# ============================================
# KoBERT 실시간 문장 예측 (안정 버전, 외부 의존성 제거)
# ============================================

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 설정
# -----------------------------
MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 경로
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

# -----------------------------
# 모델 로드 함수
# -----------------------------
def load_latest_kobert():
    """
    outputs/models/ 폴더의 최신 KoBERT 모델(.pt) 로드
    """
    model_files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith("kobert") and f.endswith(".pt")],
        reverse=True
    )
    if not model_files:
        raise FileNotFoundError("학습된 KoBERT 모델(.pt)을 찾을 수 없습니다.")

    model_path = os.path.join(MODEL_DIR, model_files[0])
    print(f" 최신 KoBERT 모델 로드 완료: {model_path}")

    # AutoTokenizer 사용 (kobert_tokenizer 대체)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,  # KoBERT는 slow tokenizer가 더 안정적
        do_lower_case=False
    )

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return tokenizer, model

# -----------------------------
# 예측 함수
# -----------------------------
def predict_text_kobert(text, tokenizer, model):
    """
    KoBERT를 사용한 문장 예측 (정상 / 보이스피싱)
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    ).to(device)

    # token_type_ids 제거 (KoBERT는 단문 기준)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100

    label = " 보이스피싱 의심" if pred == 1 else "정상 통화"
    print(f"\n 입력 문장: {text}")
    print(f"예측 결과: {label} ({confidence:.2f}% 확률)")

    if pred == 1 and confidence >= 50:
        print(" KoBERT 모델에서 높은 위험도를 감지했습니다.\n")

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    tokenizer, model = load_latest_kobert()

    while True:
        text = input("문장을 입력하세요 (종료: exit): ").strip()
        if text.lower() == "exit":
            print(" 종료합니다.")
            break
        predict_text_kobert(text, tokenizer, model)
