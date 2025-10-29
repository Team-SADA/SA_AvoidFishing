#추후 판별을 위한 단어 리스트 제작

# file: preprocessing/make_danger_word.py

import pandas as pd
from collections import Counter
import re
import os

# 데이터 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "danger_words.txt")

def clean_text(text):
    text = str(text)
    text = re.sub(r"[^가-힣0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_danger_wordlist(threshold=3):
    print("📘 위험 단어 리스트 생성 중...")
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].apply(clean_text)

    # 보이스피싱(라벨=1) 텍스트만 사용
    phishing_texts = df[df["label"] == 1]["text"]

    # 단어 토큰화
    words = []
    for text in phishing_texts:
        words.extend(text.split())

    # 단어 빈도 계산
    counter = Counter(words)
    danger_words = [word for word, count in counter.items() if count >= threshold]

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for w in sorted(danger_words):
            f.write(w + "\n")

    print(f"✅ 위험 단어 {len(danger_words)}개 저장 완료 → {OUTPUT_PATH}")
    return danger_words

if __name__ == "__main__":
    make_danger_wordlist(threshold=3)
