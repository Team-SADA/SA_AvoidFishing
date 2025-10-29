좋아요. 아래는 양식 없이 **그냥 텍스트 그대로 복사해서 remade.md**로 저장할 수 있는 형태입니다.
형식보다 내용 전달 중심으로 정리했어요 👇

---

Voice Phishing Detection Project (TF-IDF + KoBERT Hybrid)

이 프로젝트는 한국어 보이스피싱 대화문 텍스트를 기반으로 인공지능이 자동으로 보이스피싱 여부를 판별하는 시스템을 개발하는 것을 목표로 한다.
두 가지 모델을 함께 사용하여 속도와 정확성을 모두 확보하는 것이 핵심이다.

1차 판별 단계에서는 TF-IDF(Logistic Regression 기반)를 사용하여 빠르게 예측한다.
2차 판별 단계에서는 KoBERT(딥러닝 기반 문장 분류 모델)를 사용하여 TF-IDF 확률이 50% 이상일 때만 추가 판별을 수행한다.
최종 예측 확률은 TF-IDF:KoBERT = 3:2(0.6:0.4) 비율로 가중 평균하여 계산한다.

---

[코드 실행 순서]

1. 데이터 전처리 및 정제
   python preprocessing/generated_cleaned_dataset.py

2. 피처(feature) 생성 (TF-IDF 및 통계 특징)
   python preprocessing/features.py

3. TF-IDF 학습
   python models/train_tfidf.py

4. KoBERT 학습
   python models/train_kobert.py

5. 통합 예측 (하이브리드 시스템)
   python main.py

---

[디렉토리 구조]

data/raw → 원본 데이터
data/processed → 정제된 데이터 및 위험 단어 리스트
preprocessing → 데이터 정제, 토큰화, 피처 생성 코드 포함
models → TF-IDF 및 KoBERT 학습·예측·평가 코드
outputs/models → 학습된 모델 저장 위치
outputs/tokenizer → TF-IDF 벡터 저장 위치
main.py → TF-IDF와 KoBERT 결합 예측 코드
remade.md → 프로젝트 설명 문서

---

[모델 요약]

TF-IDF + Logistic Regression

* scikit-learn 사용
* 빠른 1차 탐지용

KoBERT (skt/kobert-base-v1)

* transformers 라이브러리 사용
* 문맥 기반 딥러닝 모델
* TF-IDF 50% 이상일 때만 실행

Hybrid Fusion

* TF-IDF와 KoBERT의 확률을 3:2로 가중 평균
* 최종 확률을 기준으로 판별

---

[주요 설정값]

TF-IDF
max_features = 5000
model = LogisticRegression(max_iter=1000)

KoBERT
MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 96
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

---

[작동 흐름 요약]

입력 문장 → TF-IDF 예측 →
(확률 < 50%) → 정상 판정
(확률 ≥ 50%) → KoBERT 추가 판별 →
3:2 비율로 가중 평균 → 최종 결과 출력

---

[출력 예시]

📞 문장을 입력하세요: 고객님 계좌가 범죄에 연루되었습니다
TF-IDF 예측 확률: 62.83%
KoBERT 예측 확률: 79.54%
최종 판별 결과: ⚠️ 보이스피싱 의심 (69.91% 확률)
🚨 추가 검증 권장: 고위험 문장으로 감지됨

---

[프로젝트 출처 및 참고 레포지토리]

이 프로젝트는 다음 깃허브 프로젝트를 기반으로 재구성되었다.
원본 출처: [https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts?utm_source=chatgpt.com](https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts?utm_source=chatgpt.com)

---

[리메이드 요약]

* 한국어 텍스트 전처리 및 불용어 제거 강화
* TF-IDF + KoBERT 하이브리드 구조 완성
* 위험 단어 기반 확률 가중 시스템 추가
* 실시간 예측 CLI 기능 구현
* Flask API 확장 가능 구조 설계

---

© 2025 Voice Phishing AI Defense Project
Rebuilt with GPT-5 and SKT KoBERT.

---

