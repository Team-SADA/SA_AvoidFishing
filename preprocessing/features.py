# file: preprocessing/features.py

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

# ------------------------------------------------------
#  숫자 및 키워드 기반 특징 추출 함수
# ------------------------------------------------------
def extract_text_features(texts: list[str]) -> pd.DataFrame:
    """
    텍스트 기반 통계적 특징(feature) 추출
    """
    money_keywords = ["계좌", "송금", "입금", "출금", "대출", "이체", "카드", "통장", "은행", "보내"]
    action_keywords = ["확인", "입력", "전달", "전화", "연락", "응답", "클릭", "등록", "입력하세요"]

    features = {
        "text_length": [],
        "num_words": [],
        "num_count": [],
        "money_keyword": [],
        "action_keyword": []
    }

    for text in texts:
        text = str(text)
        features["text_length"].append(len(text))
        features["num_words"].append(len(text.split()))
        features["num_count"].append(len(re.findall(r"\d+", text)))

        features["money_keyword"].append(sum(kw in text for kw in money_keywords))
        features["action_keyword"].append(sum(kw in text for kw in action_keywords))

    return pd.DataFrame(features)


# ------------------------------------------------------
#  TF-IDF 벡터화 함수
# ------------------------------------------------------
def get_tfidf_features(texts: list[str], max_features: int = 3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f" TF-IDF 피처 생성 완료 ({tfidf_matrix.shape[1]}차원)")
    return tfidf_matrix, vectorizer


# ------------------------------------------------------
#  KoBERT 임베딩 추출 함수
# ------------------------------------------------------
def get_kobert_embeddings(texts: list[str], model_name="skt/kobert-base-v1", max_len=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] 토큰 벡터
        embeddings.append(cls_vector[0])

    embeddings = np.array(embeddings)
    print(f" KoBERT 임베딩 생성 완료 ({embeddings.shape})")
    return embeddings


# ------------------------------------------------------
#  전체 피처 결합 함수
# ------------------------------------------------------
def build_features(texts: list[str], use_kobert=True):
    """
    모든 피처를 결합하여 X(feature matrix) 생성
    """
    # 텍스트 기반 통계 피처
    df_features = extract_text_features(texts)

    # TF-IDF 피처
    tfidf_matrix, _ = get_tfidf_features(texts)

    # KoBERT 임베딩 (선택)
    if use_kobert:
        bert_features = get_kobert_embeddings(texts)
        X = np.concatenate([tfidf_matrix.toarray(), bert_features, df_features.values], axis=1)
    else:
        X = np.concatenate([tfidf_matrix.toarray(), df_features.values], axis=1)

    print(f" 전체 피처 결합 완료: shape={X.shape}")
    return X, df_features.columns.tolist()


# ------------------------------------------------------
# 5️⃣ 실행 테스트
# ------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "안녕하세요, 검찰청입니다. 귀하의 계좌가 범죄에 연루되었습니다.",
        "오늘 날씨가 참 좋네요!",
        "대출 승인을 위해 인증번호 1234를 입력하세요."
    ]

    X, feature_names = build_features(sample_texts, use_kobert=False)
    print("✅ Feature matrix shape:", X.shape)
    print("📊 Features:", feature_names)
