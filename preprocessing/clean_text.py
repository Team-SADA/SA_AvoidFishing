import re
import pandas as pd

# ------------------------------------------------------
# 불용어 설정
# ------------------------------------------------------
def load_stopwords():
    # 한국어 기본 불용어 + 조사/접속사
    return [
        "은", "는", "이", "가", "을", "를", "에", "에서", "으로",
        "의", "도", "로", "와", "과", "한", "하다", "것", "거",
        "그리고", "하지만", "그래서", "또한", "입니다", "합니다"
    ]


# ------------------------------------------------------
# ⃣ 핵심 텍스트 정제 함수
# ------------------------------------------------------
def clean_text(text):
    """
    입력된 문장을 정제하여 불필요한 요소를 제거하고
    금융 키워드를 보존한 정제된 텍스트를 반환한다.
    """

    if not isinstance(text, str):
        text = str(text)

    # 1. 소문자 변환
    text = text.lower()

    # 2. 한글, 숫자, 공백, 일부 기호만 남기기
    text = re.sub(r"[^가-힣0-9\s.,?!]", " ", text)

    # 3. 중복 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    # 4. 숫자 치환
    text = re.sub(r"\d{3,}", "<NUM>", text)

    # 5. 금융 키워드 보존: (보존용 단어는 그대로 둠)
    preserve_keywords = [
        "계좌", "은행", "송금", "대출", "환급", "수사", "검찰", "경찰",
        "사건", "보안", "인증", "비밀번호", "입금", "출금", "예금",
        "카드", "통장", "대포통장", "명의", "범죄", "세금", "청구", "고객센터"
    ]
    for word in preserve_keywords:
        text = re.sub(fr"({word})", word, text)

    # 6. 불용어 제거
    stopwords = load_stopwords()
    for sw in stopwords:
        text = re.sub(fr"\b{sw}\b", "", text)

    # 7. 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------
#  전체 데이터셋 정제
# ------------------------------------------------------
def clean_dataset(df):
    """
    pandas DataFrame 전체에 clean_text 적용
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # 너무 짧은 문장은 제거 (5자 이하)
    df = df[df["clean_text"].apply(lambda x: len(x) > 5)]

    return df


# ------------------------------------------------------
#  실행 테스트용 (직접 실행 시)
# ------------------------------------------------------
if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "text": [
            "안녕하세요, 검찰청입니다. 귀하의 계좌가 범죄에 연루되었습니다!!!",
            "오늘 날씨 좋네요 ^^",
            "국세청 환급금이 있습니다. 인증번호 1234를 입력하세요.",
            "네."
        ]
    })

    cleaned_df = clean_dataset(sample_df)
    print(cleaned_df)
