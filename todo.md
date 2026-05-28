## 전체 진행 순서

```text
1. 3개년 변수명/라벨 비교
2. 공통 변수 후보 선정
3. 타깃 y 정의
4. 누수 변수, 조사설계 변수, 결측률 높은 변수 제거
5. 연도별 EDA
6. 3개년 통합 데이터셋 생성
7. 통합 EDA
8. Baseline 모델 학습
9. Stratified 10-fold cross validation으로 성능 평가
10. 성능 개선 실험
11. 최종 모델 선정
12. SHAP으로 최종 모델 해석
13. 보고서/발표 정리
```

---

### 1단계 : 변수명/라벨 비교

만들어야할 파일

```text
variables_by_year.xlsx
common_variables.xlsx
candidate_label_compare.xlsx
different_label_variables.xlsx
```

- variables_by_year.xlsx : 연도별 전체 변수 목록 확인
- common_variables.xlsx : 3개년에 모두 존재하는 변수 확인
- candidate_label_compare.xlsx : 우리가 쓰려는 후부 변수의 라벨 동일성 확인
- different_label_variables.xlsx : 변수명은 같지만 라벨이 다른 변수 확인

```text
연도별 문항 차이를 고려하여 2023~2025 자료에서 공통으로 존재하는 변수만 후보 변수로 선정하였다.
이후 변수 라벨을 비교하여 동일한 의미의 문항인지 확인하였다. 
```

---

### 2단계 : 타깃 y 정의

`현재 액상형 전다담배 사용 여부`

`타깃 변수는 최근 30일 동안 액상형 전자담배를 사용했는지 여부로 정의하였다.
원자료의 응답값을 바탕으로 사용하지 않은 경우 0, 1일 이상 사용한 경우 1로 이진화하였다.`

---

### 3단계 : 제거할 feature 정리

#### 반드시 제거할 변수

1. 타깃 누수 변수

```text
TC_EC_MN
TC_EC_LT
TC_EC_FAGE
액상형 전자담배 사용량/사용 이유 관련 변수
```

현재 액상형 전자담배 사용 여부를 예측하는데 액상형 전다담배 평생 경험이나 최초 사용 시기를 넣으면 정답 힌드를 주는 셈

```text
타깃 변수와 직접적으로 중복되거나 사후적으로 알 수 있는 전자담배 관련 변수는 데이터 누수를 방지하기 위해 제거하였다.
```

2. 조사설계/식별자 변수

```text
OBS
CLUSTER
STRATA
STRATA_NM
GROUP
FPC
W
mod_d
```

이 변수들은 개인의 행동·환경 요인을 설명하는 feature가 아님

3. 결측률이 높은 변수

```text
결측률 30% 이상: 제거 검토
결측률 50% 이상: 제거
```

---

### 4단계: 최종 feature 후보

처음부터 너무 많이 넣지 말고, 20~30개 정도로 시작

그룹

- 기본 배경
- 정신 건강
- 음주
- 식습관
- 신체활동
- 스마트폰/생활
- 가족환경

---

### 5단계 : EDA는 연도별 + 통합 둘 다

EDA : 탐색적 데이터 분석

#### 연도별 EDA

목적 : 3개년 데이터를 합쳐도 되는지, 연도별 분포 차이가 있는지 확인  
볼 것 :

```text
각 연도 표본 수
각 연도 y 비율
연도별 액상형 전자담배 현재사용률
주요 feature 분포
결측률 차이
```

#### 통합 EDA

목적 : 최종 모델이 학습할 데이터의 구조 확인  
볼 것 :

```text
전체 y 클래스 분포
feature 간 상관관계 heatmap
주요 변수별 y 비율
불균형 정도
```

---

### 6단계 : Baseline 모델 만들기

기본 모델 세 개로 시작 하기

```text
Logistic Regression
Decision Tree
Random Forest
```

#### 각 모델 역할

| 모델                  | 역할          |
|---------------------|-------------|
| Logistic Regression | 기준 성능 비교 모델 |
| Decision Tree       | 규칙 기반 해석 모델 |
| Random Forest       | 주력 모델 후보    |


보고서 문장 :
```text
먼저 기본 전처리만 적용한 상태에서 Logistic Regression, Decision Tree, Random Forest를 학습하여
기준 성능을 확인하였다. 이후 성능 개선 실험을 통해 모델의 예측 성능 변화를 비교하였다.
```  
---

### 7단계 : Stratified 10-fold cross validation으로 평가 
전자 담배 사용자가 적을 가능성이 높으니 일반 KFold보다 **StratifiedKFold** 를 사용해보자  
StratifiedKFold는 각 fold에서 클래스 비율을 보존하도록 나누는 방식  

보고서 문장 : 
```text
단일 train/test split에 따른 성능 편차를 줄이기 위해 Stratified 10-fold crossvalidation을 적용하였다.
타깃 클래스 불균형을 고려하여 각 fold에서 사용/비사용 클래스 비율이 유사하게 유지되도록 하였다.
```

---
### 8단계 : 평가 지표
Accuracy만 보면 안된다.  

전자 담배 사용자가 전체의 3% 정도라면, 모델이 전부 "비사용"이라고 예측해도 Accuracy는 높게 나올 수 있음  

| 지표        | 의미                        |
|-----------|---------------------------|
| Accuracy  | 전체 예측 중 맞힌 비율             |
| Precision | 사용한다고 예측한 학생 중 실제 사용자의 비율 |
| Recall    | 실제 사용자 중 모델이 찾아낸 비율       |
| F1-score  | Precision과 Recall의 균형     |
| ROC-AUC   | 전체적인 구분 능력                |

최종 모델은 **Accuracy 최고 모델**이 아니라, **F1, Recall, ROC_AUC, 해석 가능성**을 함께 보고 고르자  

---

### 9단계 : 성능 개선 실험 
성능이 낮거나 Recall이 너무 낮으면 개선 실험을 해야 함  

#### 실험 A. Baseline  
불균형 처리 없이 기본 모델 학습  
목적 : 기준 성능 확인  

#### 실험 B. class_weight='balanced'
데이터 개수를 바꾸지 않고, 소수 클래스의 오분류 비용을 더 크게 주는 방식  

장점 : 
- 구현 쉬움
- 데이터 자체를 합성하지 않음 
- 보고서에서 설명하기 쉬움  

#### 실험 C. SMOTE 또는 SMOTENC
"부족한 y 데이터와 비슷한 데이터를 추가한다"  
주의할 점 : 
```text
SMOTE를 전체 데이터에 먼저 적용한 뒤 train/test split이나 cross-validation을 하면
data leakage가 생길 수 있음. imbalanced-learn 공식 문서도 전체 데이터에 resampling을 먼저 적용하고 나누는  
것을 흔한 실수라고 설명함  
```

SMOTE는 반드시 : 
- 각 fold의 train 데이터에만 적용 
- validation/test 데이터에는 적용하지 않음  

실험표:

| 실험            | 처리 방식                   | 목적               |
|---------------|-------------------------|------------------|
| Baseline      | 불균형 처리 없음               | 기준 성능 확인         |
| Class weight  | class_weight='balanced' | 소수 클래스 오분류 비용 증가 |
| SMOTE/SMOTENC | 학습 fold에만 오버샘플링         | 소수 클래스 데이터 보강    |
| Tuning        | 하이퍼파라미터 조정              | 최종 성능 개선         |

---

### 10단계 : 하이퍼파라미터 튜닝
기본 모델 성능이 낮으면 튜닝해야 함 

Decision Tree : 
```text
max_depth
min_samples_leaf
min_samples_split
```

Random Forest:  
```text
n_estimators
max_depth
min_samples_leaf
max_features
class_weight
```

보고서 문장 : 
```text
모델의 일반화 성능을 높이기 위해 주요 하이퍼파라미터를 조정하였다. Decision Tree와 
Random Forest에서는 최대 깊이, 최소 리프 샘플 수, 트리 개수 등을 중심으로 탐색하였다.
```

---
### 11단계 : 최종 모델 선정 
우선 순위 : 
```text
1. ROC-AUC가 높은가?
2. F1-score가 좋은가?
3. Recall이 너무 낮지 않은가?
4. Confusion Matrix에서 y=1을 어느 정도 찾아내는가?
5. SHAP으로 설명 가능한가?
6. 모델이 너무 복잡하지 않은가?
```
보고서 문장 : 
```text
최종 모델은 Accuracy만을 기준으로 선정하지 않고, 클래스 불균형을 고려하여 F1-score, Recall, 
ROC-AUC를 함께 검토하여 선정하였다. 
```

---
## 12단계 : SHAP으로 최종 모델 설명 
SHAP은 **최종 모델을 고른 다음** 적용  

순서 : 
```text
모델 비교
↓
성능 개선 실험
↓
최종 모델 선정
↓
SHAP 분석
```

```text
1. SHAP bar plot
2. SHAP summary plot
```
SHAP bar plot은 전체적으로 중요한 변수를 보여주고, summary plot은 변수값이 예측을  
높이는 방향인지 낮추는 방향인지 보여줄 수 있음  

보고서 문장 : 
```text
최종 모델의 예측 과정을 설명하기 위해 SHAP 분석을 수행하였다. 
SHAP 값은 각 feature가 모델 예측값을 높이거나 낮추는 데 기여한 정도를 나타내며,  
본 연구에서는 이를 인과관계가 아닌 예측 기여도로 해석하였다.  
```

---
### 최종 보고서 구조 
```text
1. 서론
   - 문제 배경
   - 프로젝트 목표

2. 데이터 설명
   - 데이터 출처
   - 사용 연도
   - 타깃 변수 정의
   - 후보 feature 설명

3. 데이터 전처리
   - 공통 변수 선정
   - 라벨 동일성 확인
   - 결측/무응답 처리
   - 누수 변수 제거
   - 인코딩/스케일링

4. 탐색적 데이터 분석
   - 연도별 사용률
   - 클래스 분포
   - 주요 변수별 사용률
   - 상관관계 heatmap

5. 모델 학습 및 검증
   - Baseline 모델
   - Stratified 10-fold cross validation
   - 평가 지표

6. 성능 개선 실험
   - class_weight
   - SMOTE/SMOTENC
   - 하이퍼파라미터 튜닝
   - 개선 전후 비교

7. 최종 모델 해석
   - Feature Importance
   - Permutation Importance
   - SHAP 분석

8. 결론 및 한계
   - 최종 모델
   - 주요 예측 기여 변수
   - 한계점
   - 향후 개선 방향
```
---
### 최종 발표 구조 
```text
1. 주제 및 문제의식
2. 데이터 소개
3. 분석 목표
4. 전체 분석 프로세스
5. 변수 선정 및 제거 기준
6. EDA 결과
7. 모델링 방법
8. 10-fold CV 성능 비교
9. 성능 개선 실험 결과
10. SHAP 분석 결과
11. 결론 및 시사점
12. 한계 및 향후 개선
```

