import os
import pandas as pd
from preprocessing.features import extract_text_features

# 프로젝트 루트 기준 절대 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "KorCCViD_v1.3_fullcleansed.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")

# processed 폴더가 없으면 자동 생성
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(RAW_PATH)
print(df.columns)

# 컬럼명 통일
df.rename(columns={'Transcript': 'text', 'Label': 'label'}, inplace=True)

# 텍스트 정제
df["text"] = df["text"].astype(str).str.replace(r"[^가-힣0-9\s]", "", regex=True)

# 피처 생성 테스트
features = extract_text_features(df["text"])
print(" Features created:", features.shape)

# 저장
df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
print(f" 저장 완료: {PROCESSED_PATH}")
