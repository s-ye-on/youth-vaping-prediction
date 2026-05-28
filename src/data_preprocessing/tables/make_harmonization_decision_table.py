import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR

OUTPUT_DIR = TABLES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "harmonization_decision_table.xlsx"

rows = [
    {
        "variable": "E_BORN_F",
        "decision": "use",
        "reason": "2023~2024년은 '아버지 태어난 나라', 2025년은 '아버지 출신 국적'으로 라벨 표현은 다르지만, 3개년 모두 unique_count=15, 값 범위 1~9999로 동일하여 부모 출신 배경 계열 변수로 통합 가능하다고 판단하였다.",
        "caution": "9999는 정상 국가 코드가 아니라 비해당/무응답 계열 값일 가능성이 있으므로 별도 범주 또는 결측 처리 방식을 기록한다."
    },
    {
        "variable": "E_BORN_M",
        "decision": "use",
        "reason": "2023~2024년은 '어머니 태어난 나라', 2025년은 '어머니 출신 국적'으로 라벨 표현은 다르지만, 3개년 모두 unique_count=15, 값 범위 1~9999로 동일하여 부모 출신 배경 계열 변수로 통합 가능하다고 판단하였다.",
        "caution": "9999는 정상 국가 코드가 아니라 비해당/무응답 계열 값일 가능성이 있으므로 별도 범주 또는 결측 처리 방식을 기록한다."
    },
    {
        "variable": "F_BR",
        "decision": "use",
        "reason": "아침식사 빈도/횟수로 라벨 표현은 다르지만, 3개년 모두 unique_count=8, 값 범위 1~8로 동일하여 동일한 식습관 변수로 판단하였다.",
        "caution": "순서형 설문 변수로 처리한다."
    },
    {
        "variable": "F_FASTFOOD",
        "decision": "use",
        "reason": "패스트푸드 섭취빈도/섭취 횟수로 라벨 표현은 다르지만, 3개년 모두 unique_count=7, 값 범위 1~7로 동일하여 동일한 식습관 변수로 판단하였다.",
        "caution": "순서형 설문 변수로 처리한다."
    },
    {
        "variable": "F_FRUIT",
        "decision": "use",
        "reason": "과일 섭취빈도/섭취 횟수로 라벨 표현은 다르지만, 3개년 모두 unique_count=7, 값 범위 1~7로 동일하여 동일한 식습관 변수로 판단하였다.",
        "caution": "순서형 설문 변수로 처리한다."
    },
    {
        "variable": "F_SWD_A",
        "decision": "use",
        "reason": "단맛 나는 음료 섭취빈도/섭취 횟수로 라벨 표현은 다르지만, 3개년 모두 unique_count=7, 값 범위 1~7로 동일하여 동일한 식습관 변수로 판단하였다.",
        "caution": "순서형 설문 변수로 처리한다."
    },
    {
        "variable": "F_WAT",
        "decision": "use",
        "reason": "2025년 라벨은 '물 섭취량'으로 표현되었으나, 3개년 모두 unique_count=5, 값 범위 1~5로 동일하여 동일 계열 변수로 통합 가능하다고 판단하였다.",
        "caution": "라벨 표현 차이가 있으므로 보고서에 정합화 기준을 명시한다."
    },
    {
        "variable": "M_SAD",
        "decision": "use",
        "reason": "슬픔&절망감 경험/우울감 경험으로 라벨 표현은 다르지만, 3개년 모두 unique_count=2, 값 범위 1~2로 동일하여 동일한 정신건강 계열 변수로 판단하였다.",
        "caution": "민감 변수이므로 인과관계가 아닌 예측 기여도로만 해석한다."
    },
    {
        "variable": "CITY",
        "decision": "use",
        "reason": "3개년 모두 17개 시도 값을 가지며, 외부 지역 단위 변수와 결합하기 위한 기준 변수로 필요하다.",
        "caution": "지역 변수는 개인의 원인으로 해석하지 않는다."
    },
    {
        "variable": "CTYPE",
        "decision": "use",
        "reason": "3개년 모두 도시규모 변수이며 unique_count=3으로 동일하다.",
        "caution": "표본설계용 CTYPE_SD와 구분하여 사용한다."
    },
]

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print("저장 완료:", OUTPUT_FILE)
print(df.to_string(index=False))
