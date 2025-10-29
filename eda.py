import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    print("✅ 데이터 로드 완료")
    print("📄 컬럼명:", df.columns.tolist())

    # 컬럼명 정리 (공백 제거)
    df.columns = df.columns.str.strip()

    # 텍스트 컬럼명 자동 탐색 (Transcript도 포함)
    text_col_candidates = [col for col in df.columns if any(x in col.lower() for x in ["text", "sentence", "transcript"])]
    if len(text_col_candidates) == 0:
        raise ValueError("❌ 텍스트 컬럼을 찾을 수 없습니다. CSV 파일의 열 이름을 확인하세요.")
    text_col = text_col_candidates[0]
    print(f"🧠 텍스트 컬럼 인식됨: {text_col}")

    # 통합 text 컬럼 생성
    df["text"] = df[text_col].astype(str)

    # 라벨 이름 통일
    if "label" not in df.columns:
        label_col_candidates = [col for col in df.columns if "label" in col.lower()]
        if len(label_col_candidates) == 0:
            raise ValueError("❌ 라벨 컬럼을 찾을 수 없습니다.")
        label_col = label_col_candidates[0]
        df.rename(columns={label_col: "label"}, inplace=True)

    return df

def analyze_data(df):
    df["length"] = df["text"].apply(lambda x: len(str(x)))
    print("\n📊 평균 문장 길이:", df["length"].mean())
    print("라벨 분포:\n", df["label"].value_counts())

    df["label"].value_counts().plot(kind="bar", title="Label Distribution")
    plt.show()

    df["length"].plot(kind="hist", bins=50, title="Text Length Distribution")
    plt.show()

if __name__ == "__main__":
    df = load_data("data/raw/KorCCViD_v1.3_fullcleansed.csv")
    analyze_data(df)
