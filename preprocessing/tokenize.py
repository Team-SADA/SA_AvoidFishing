# file: preprocessing/tokenize.py

import re
from typing import List
from transformers import AutoTokenizer
import nltk

# NLTK 다운로드 (처음 실행 시 자동 설치)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ------------------------------------------------------
# 1️⃣ KoBERT용 AutoTokenizer
# ------------------------------------------------------
def get_kobert_tokenizer(model_name: str = "skt/kobert-base-v1"):
    """
    KoBERT용 SentencePiece 기반 AutoTokenizer 로드
    (protobuf 및 캐시 오류 처리 포함)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ KoBERT tokenizer 로드 완료: {model_name}")
        return tokenizer

    except ImportError as e:
        print("❌ Protobuf 관련 오류가 발생했습니다.")
        print("🔧 해결 방법:")
        print("  1️⃣ pip uninstall protobuf -y")
        print("  2️⃣ pip install protobuf==3.20.3")
        print("  3️⃣ 가상환경 재시작 후 다시 실행")
        raise e

    except Exception as e:
        print(f"❌ KoBERT tokenizer 로드 실패: {e}")
        raise e


# ------------------------------------------------------
# 2️⃣ KoBERT 입력 변환 함수
# ------------------------------------------------------
def tokenize_for_kobert(texts: List[str], tokenizer, max_len: int = 64):
    """
    KoBERT 모델에 맞게 텍스트를 토큰화하여
    PyTorch tensor 형식으로 반환합니다.
    """
    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"  # PyTorch tensor로 반환
    )

    return encoded


# ------------------------------------------------------
# 3️⃣ TF-IDF용 간단 토큰화 함수
# ------------------------------------------------------
def tokenize_for_tfidf(texts: List[str]) -> List[str]:
    """
    TF-IDF 입력용 단순 단어 기반 토큰화
    (한글과 숫자만 유지, 띄어쓰기 기준)
    """
    tokenized_texts = []
    for text in texts:
        cleaned = re.sub(r"[^가-힣0-9\s]", " ", str(text))  # 한글/숫자 외 제거
        tokens = cleaned.split()  # 띄어쓰기 기준 토큰화
        tokens = [t for t in tokens if len(t) > 1]  # 한 글자 제거
        tokenized_texts.append(" ".join(tokens))
    return tokenized_texts


# ------------------------------------------------------
# 4️⃣ MeCab 형태소 분석 기반 토큰화 (선택적)
# ------------------------------------------------------
def tokenize_with_mecab(texts: List[str], mecab=None) -> List[str]:
    """
    형태소 분석 기반 한국어 토큰화 (선택적)
    """
    if mecab is None:
        from konlpy.tag import Mecab
        mecab = Mecab()

    tokenized_texts = []
    for text in texts:
        tokens = mecab.morphs(text)
        tokenized_texts.append(" ".join(tokens))
    return tokenized_texts


# ------------------------------------------------------
# 5️⃣ 실행 테스트 (단독 실행 시)
# ------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "안녕하세요, 검찰청입니다. 귀하의 계좌가 범죄에 연루되었습니다.",
        "오늘 날씨가 참 좋네요!",
        "대출 승인을 위해 인증번호 1234를 입력하세요."
    ]

    # KoBERT 토크나이저 테스트
    try:
        tokenizer = get_kobert_tokenizer()
        encoded = tokenize_for_kobert(sample_texts, tokenizer)
        print("\n🧠 KoBERT Tokenized Example:")
        print(encoded["input_ids"][0][:10])  # 일부만 출력
    except Exception as e:
        print("\n⚠️ KoBERT 토크나이저 테스트 실패:", e)

    # TF-IDF 토크나이저 테스트
    print("\n⚡ TF-IDF Tokenized Example:")
    tfidf_tokens = tokenize_for_tfidf(sample_texts)
    for t in tfidf_tokens:
        print("-", t)

    # MeCab 테스트 (선택)
    try:
        mecab_tokens = tokenize_with_mecab(sample_texts)
        print("\n🪄 MeCab Tokenized Example:")
        for t in mecab_tokens:
            print("-", t)
    except Exception:
        print("\n⚠️ MeCab이 설치되어 있지 않습니다. 형태소 분석 스킵.")
