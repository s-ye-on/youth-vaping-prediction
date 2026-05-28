# 라벨이 달랐던 8개 변수와 주요 변수에 대해 통합 여부를 표로 정리

# decide_harmonized_variables.py

import pandas as pd

from src.data_preprocessing.project_paths import TABLES_DIR

rows = [
    {
        "variable": "E_BORN_F",
        "label_issue": "2023/2024: 아버지 태어난 나라, 2025: 아버지 출신 국적",
        "decision": "use",
        "reason": "부모의 출신 배경을 나타내는 동일 계열 변수로 판단. 실제 값 범위와 분포 확인 후 통합.",
        "caution": "엄밀히는 출생국과 국적 표현 차이가 있으므로 보고서에 정합화 기준 명시."
    },
    {
        "variable": "E_BORN_M",
        "label_issue": "2023/2024: 어머니 태어난 나라, 2025: 어머니 출신 국적",
        "decision": "use",
        "reason": "부모의 출신 배경을 나타내는 동일 계열 변수로 판단. 실제 값 범위와 분포 확인 후 통합.",
        "caution": "엄밀히는 출생국과 국적 표현 차이가 있으므로 보고서에 정합화 기준 명시."
    },
    {
        "variable": "F_BR",
        "label_issue": "아침식사 빈도 vs 아침식사 횟수",
        "decision": "use_if_value_distribution_consistent",
        "reason": "최근 7일 동안의 아침식사 빈도/횟수는 사실상 같은 생활습관 지표로 판단 가능.",
        "caution": "연도별 값 범위와 빈도표 확인 필요."
    },
    {
        "variable": "F_FASTFOOD",
        "label_issue": "패스트푸드 섭취빈도 vs 섭취 횟수",
        "decision": "use_if_value_distribution_consistent",
        "reason": "최근 7일 동안의 패스트푸드 섭취 빈도/횟수는 같은 식습관 지표로 판단 가능.",
        "caution": "연도별 값 범위와 빈도표 확인 필요."
    },
    {
        "variable": "F_FRUIT",
        "label_issue": "과일 섭취빈도 vs 섭취 횟수",
        "decision": "use_if_value_distribution_consistent",
        "reason": "최근 7일 동안의 과일 섭취 빈도/횟수는 같은 식습관 지표로 판단 가능.",
        "caution": "연도별 값 범위와 빈도표 확인 필요."
    },
    {
        "variable": "F_SWD_A",
        "label_issue": "단맛 나는 음료 섭취빈도 vs 섭취 횟수",
        "decision": "use_if_value_distribution_consistent",
        "reason": "최근 7일 동안의 단맛 음료 섭취 빈도/횟수는 같은 식습관 지표로 판단 가능.",
        "caution": "연도별 값 범위와 빈도표 확인 필요."
    },
    {
        "variable": "F_WAT",
        "label_issue": "물 섭취 빈도 vs 물 섭취량",
        "decision": "review_after_distribution_check",
        "reason": "빈도와 섭취량은 다를 수 있으므로 실제 값 체계 확인 후 결정.",
        "caution": "값 범위가 다르면 제외 또는 별도 변환 필요."
    },
    {
        "variable": "M_SAD",
        "label_issue": "슬픔&절망감 경험 vs 우울감 경험",
        "decision": "use_if_value_distribution_consistent",
        "reason": "정신건강 관련 동일 계열 변수로 판단 가능.",
        "caution": "민감 변수이므로 인과관계로 해석하지 않음."
    },
    {
        "variable": "CITY",
        "label_issue": "라벨 동일",
        "decision": "use",
        "reason": "지역 단위 외부 feature 결합을 위해 필요.",
        "caution": "지역 자체가 개인 원인이라는 식으로 해석하지 않음."
    },
    {
        "variable": "CTYPE",
        "label_issue": "라벨 동일",
        "decision": "use",
        "reason": "도시규모 및 환경 요인 반영 가능.",
        "caution": "표본설계용 CTYPE_SD와 구분."
    },
]

df = pd.DataFrame(rows)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_file = TABLES_DIR / "harmonization_decision_table.xlsx"
df.to_excel(output_file, index=False)

print("저장 완료:", output_file)
print(df.to_string(index=False))
