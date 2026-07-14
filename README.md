# 반도체 FDC 자동 잔여수명(RUL) 예측 시스템

다중 센서 FDC trace에서 건강 기준 구간을 자동 설정하고, 2σ 경고 신호를 확인한 뒤 열화 변곡점을 역추적하여 활성 열화 구간만 모델링합니다. 시간 순서를 보존한 AutoML 검증으로 열화 곡선을 선택하고 센서별 3σ 고장 한계와의 최초 교차 시각을 계산합니다. 장비 RUL은 센서별 예측 중 가장 이른 고장 시각을 기준으로 결정합니다.

웹 화면, CLI, JSON, CSV, HTML, 그래프의 주요 표시는 한글입니다. 정규화 z-score는 내부 탐지에만 사용하며, 최종 그래프와 예측 CSV의 trace·평활값·임계값·복원 궤적은 모두 원시 계측 단위입니다.

## 구성

```text
src/rul_system/
├── preprocessing_and_onset.py  # 입력, EWMA, 롤링 특징, 2σ/3σ, 변곡점 탐지
├── models.py                   # 선형/다항/지수/SVR/부스팅/XGBoost 모델
├── automl.py                   # 시계열 CV, GridSearch, RMSE/R²/AIC/BIC, 외삽
├── pipeline.py                 # 센서별 예측과 장비 RUL 결정
├── reporting.py                # 원시 단위 PNG/CSV/JSON/HTML 보고서
├── webapp.py                   # FastAPI 분석 API와 웹 화면 데이터 계약
├── web/static/                 # 반응형 한글 대시보드와 Canvas 차트
├── sample_data.py              # 정상·열화·급격 고장 합성 데이터
├── config.py                   # 모듈별 교체 가능한 설정 객체
└── cli.py                      # 한글 명령행 인터페이스
```

## 설치 및 첫 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
rul-system demo --output demo_output
```

## 현업 사용자용 웹 화면

```bash
pip install -e '.[web,dev]'
rul-system web --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 열면 다음 기능을 사용할 수 있습니다.

- CSV/Parquet 드래그 앤 드롭과 정상·완만 열화·급격 고장 샘플 실행
- EWMA, 건강 기준 비율, 2σ/3σ, 연속 이상점, AutoML 후보 설정
- 설비 상태, 예상 고장일, RUL 일/시간, 주요 위험 센서 KPI
- 입력 검증부터 RUL 계산까지 6단계 분석 과정과 판정 근거
- 원시 trace, EWMA, 적합·외삽 곡선, 관리 한계, 열화 시작점
- 통합 이상점수, 최신 특징값, 모델식, RMSE/R²/AIC/BIC, AutoML 순위표
- 최근 원시 데이터 조회와 한글 요약 JSON 저장

웹 API 문서는 `http://127.0.0.1:8000/api/docs`에서 확인할 수 있습니다. 업로드 데이터는 서버 메모리에서 분석하고 별도 저장하지 않습니다.

XGBoost 후보까지 활성화하려면 다음과 같이 설치합니다. 설치되지 않은 경우에도 나머지 모델로 정상 동작합니다.

```bash
pip install -e '.[xgboost,dev]'
```

실제 CSV 분석:

```bash
rul-system analyze fdc_trace.csv \
  --timestamp EventTime \
  --sensors ChamberPressure RFMatch CoolantFlow \
  --output rul_output
```

Parquet도 같은 방식으로 입력할 수 있습니다. Parquet 입력이 필요하면 `pip install -e '.[parquet]'`로 엔진을 함께 설치합니다. 센서 목록을 생략하면 시간 열을 제외한 숫자형 열을 자동 선택합니다.

## 입력 데이터 계약

| 항목 | 요구사항 |
|---|---|
| 시간 열 | `pandas.to_datetime`으로 변환 가능한 값. 기본 이름은 `timestamp` |
| 센서 열 | 숫자로 변환 가능한 FDC trace. 여러 센서 지원 |
| 정렬/중복 | 시간순 자동 정렬, 동일 시각 중복은 원시 단위 평균 집계 |
| 결측 | 원시 열에는 결측을 유지하고 내부 `__clean` 열에서 제한 보간 |
| 최소 길이 | 전처리 8개 이상. 안정적인 기준/AutoML에는 30개 이상 권장 |

## 분석 흐름

1. 중앙값 필터와 EWMA로 노이즈를 줄입니다.
2. 롤링 평균·표준편차, 시간당 기울기, 누적 변화량을 자동 생성합니다.
3. 초기 건강 기준의 원시 평균/표준편차로 방향별 2σ 경고와 3σ 고장 한계를 계산합니다.
4. 2σ를 설정된 횟수만큼 연속 통과하면 이상을 확인하고, 기준 상태와 열화 회귀의 분할 오차가 최소인 변곡점까지 역추적합니다.
5. 변곡점 이후만 선형, 2/3차 다항, 지수, SVR, Gradient Boosting 및 선택적 XGBoost로 적합합니다.
6. 확장형 시계열 교차검증 RMSE/R²와 복잡도·외삽 방향 페널티를 합쳐 모델을 선택하며 AIC/BIC도 기록합니다.
7. 선택 곡선이 3σ 고장 한계와 처음 교차하는 시각을 수치적으로 구해 `YYYY-MM-DD`, RUL 일/시간으로 변환합니다.

## 출력

- `rul_summary.json`: 한글 장비/센서 결과, 모델 파라미터, RMSE, R², AIC, BIC, AutoML 순위표
- `rul_diagnostic.png`: 원시 trace, EWMA, 변곡점, 적합/외삽 곡선, 2σ/3σ 한계. 수식은 그래프 내부, 범례는 그래프 밖에 배치
- `forecast_<sensor>.csv`: 원시 센서 단위의 외삽 궤적과 관리 한계
- `rul_report.html`: 엔지니어 전달용 한글 요약 보고서

## 설정

기본 설정 파일을 만들고 수정할 수 있습니다.

```bash
rul-system init-config --path rul_config.json
rul-system analyze fdc_trace.csv --config rul_config.json --output rul_output
```

주요 설정은 `baseline_fraction`, `warning_sigma=2.0`, `failure_sigma=3.0`, `consecutive_points`, EWMA/롤링 윈도우, AutoML 후보, 최대 외삽 일수입니다. 물리 고장 기준이 따로 있는 센서는 `onset.failure_threshold_overrides`에 `{"센서명": 원시단위_고장값}`을 지정하면 기본 3σ 대신 사용합니다.

## 예외 처리

- **드리프트 없음:** 고장일/RUL을 억지로 산출하지 않고 `산출 불가`로 반환합니다.
- **급격 고장:** 원시 신호의 큰 순간 점프와 3σ 지속 도달을 함께 확인하여 RUL 0으로 반환합니다.
- **열화 데이터 부족:** 최소 활성 구간 미달 시 외삽을 중단하고 경고를 기록합니다.
- **외삽 교차 없음:** 설정된 최대 기간 안에 3σ와 만나지 않으면 실패를 명시하고 고장일을 비워 둡니다.
- **센서별 상·하강 열화:** 건강 기준 대비 방향을 자동 추정해 UCL 또는 LCL 방향의 고장 한계를 사용합니다.

> 이 시스템은 통계적 예측 도구입니다. 생산 적용 전 설비별 물리 고장 기준, PM 이력, censoring, 운전 recipe/lot 조건과 함께 backtesting해야 합니다.

## 테스트

```bash
pytest
```

테스트는 원시값 보존, 특징 생성, 변곡점 탐지, 정상 무드리프트, 급격 고장, AutoML 외삽, 한글 보고서 생성을 검증합니다.
