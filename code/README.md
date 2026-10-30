# X5 분석 코드

* 읽는 순서: `data.ipynb` → `models.ipynb` → `holdout.ipynb` → `segments.ipynb`
* 노트북 역할: 단계별 실행 코드·저장 결과·수치 검산·그림 확인
* 기본 실행: 공개 `evidence/`로 결과 확인; 전체 재실행은 원본 CSV가 있을 때만 선택
* Python 스크립트 역할: 구매 로그 대량 처리·nested CV 반복·최종 평가의 계산 함수

| 노트북 | 확인할 내용 |
|---|---|
| [data.ipynb](notebooks/data.ipynb) | 데이터 통합·피처 생성·분할 코드, 입력 시점 점검 |
| [models.ipynb](notebooks/models.ipynb) | 후보 구성·nested CV·선정 코드, CV 평균 Qini |
| [holdout.ipynb](notebooks/holdout.ipynb) | 최종 학습·holdout 평가 코드, Qini·10분위·상위 30% 결과 |
| [segments.ipynb](notebooks/segments.ipynb) | 사후 분석 코드, 고객군 비교·SHAP 기여도 |

## 코드 찾기

| 단계 | 먼저 볼 파일 | 관련 파일 |
|---|---|---|
| 데이터·피처 | [build_features.py](modeling/build_features.py) | `add_redeem_b.py`, `split.py`, `split_final_folds.py`, `checks/` |
| 학습·선정 | [models.ipynb](notebooks/models.ipynb) | `learners.py`, `kl_filter.py`, `qini.py`, `tune.py`, `selection.py` |
| 최종 평가 | [final.py](modeling/final.py) | `holdout_eval.py` |
| 사후 분석 | [post_heterogeneity.py](modeling/post_heterogeneity.py) | `post_subgroup.py`, `post_subgroup_limits.py`, `post_score_drivers.py`, `post_metric_check.py`, `report_baseline.py`, `report_supplement.py` |
| 그림 | [holdout.ipynb](notebooks/holdout.ipynb) | Qini·10분위 그림; 고객군 그림은 `segments.ipynb` |
| 검증 | [run_all.py](modeling/tests/run_all.py) | `tests/test_*.py`, `checks/` |

## 저장 결과 찾기

| 폴더 | 먼저 볼 파일 | 나머지 내용 |
|---|---|---|
| [checks/](evidence/checks/) | `redeem_balance.out.txt` | 시점·처치 균형 점검 출력 |
| [selection/](evidence/selection/) | `qini_table.csv` | 후보 선정 기록·보조 검산표 |
| [final/](evidence/final/) | `holdout_eval/holdout_eval.json` | 최종 파라미터·봉인 기록 |
| [post_heterogeneity/](evidence/post_heterogeneity/) | `deciles.csv` | 분위·점수 구간 결과 |
| [post_subgroup/](evidence/post_subgroup/) | `contrasts.csv` | 고객군별 관측 차이 |
| [post_subgroup_limits/](evidence/post_subgroup_limits/) | `contrasts_all.csv` | 다중비교·척도·교차 검증 |
| [post_score_drivers/](evidence/post_score_drivers/) | `score_drivers.json` | SHAP 비중 |
| [post_metric_check/](evidence/post_metric_check/) | `metric_check.csv` | 선정 지표 대조 |
| [report_baseline.json](evidence/report_baseline.json) · [report_supplement.json](evidence/report_supplement.json) | 해당 JSON | README 수치의 보조 계산 |

## 입력·실행 범위

* 노트북 기본 입력: 공개 `evidence/`, 프로젝트 루트 `figs/`
* 원자료: [X5 RetailHero 공식 데이터](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) (저장소에 미포함)
* 전체 재실행 입력: `X5_RAW_DATA_DIR`의 `clients.csv`, `products.csv`, `purchases.csv`, `uplift_train.csv`
* 전체 재실행 출력: 포트폴리오 폴더 밖의 새 절대 경로 `X5_WORK_DIR`; 두 경로 지정 후 `X5_RUN_FULL=1`
* 전체 재실행 순서: `data.ipynb` → `models.ipynb` → `holdout.ipynb` → `segments.ipynb`
* 원본 미지정 시: 전체 재실행 셀 생략; 공개 집계표·그림 확인 가능
* 공개 제외: 원시 CSV, 고객별 피처·분할·예측 파일, nested CV 학습 산출물, 학습 모델
* 재실행 범위: 기본 실행은 공개 집계표 검산과 10분위·고객군 그림; 전체 학습과 Qini 곡선에는 비공개 원본 입력 필요
* 기록된 환경: Python 3.12.14, arm64 macOS. 패키지는 `pip install -r code/env/requirements.lock.txt`로 설치하며 Jupyter는 별도로 준비한다.
* 노트북은 프로젝트의 `code/notebooks/`에서 열고, 전체 실행 시 외부 원자료 및 작업 경로를 환경변수로 지정한다.
* 해석 범위: `+7.37%p`·`+3.32%p`는 holdout 관측 구매율 차이; 실제 적용 성과·매출 증가율 아님
