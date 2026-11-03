# 미니장미 잎 탐색·상태 판정 모델

스마트큐브와 농장의 장미 사진에서 잎을 찾고, 잎마다 상태를 판정하는 모델의 버전별 기록입니다.

- **이 저장소는 공개 상태입니다** (2026-11-03 확인). 공개 관련 확인 항목은 [NOTICE.md](NOTICE.md)를 보세요.
- 원본 `/helpai-storage/model`의 복사본에서 시작했고, 원본은 수정하지 않습니다.
- 전체 변경 이력: [CHANGELOG.md](CHANGELOG.md)
- 버전 비교: [REPORTS.html](REPORTS.html). 각 버전 폴더의 `REPORT.html`은 직전 기준과 비교한 리포트이고, `REPORT.md`는 요약본입니다. HTML은 내려받아 브라우저로 엽니다.

## 현재 상태 (2026-11-03)
| 항목 | 위치 |
|---|---|
| 현재 배포본 | [v9_fastinfer/deploy](v9_fastinfer/deploy/README.md) |
| 현재 모델 원본 | v8_reallabel (`leaf_seg_v8r2.pt`, `leaf_cls_v8.pt`) |
| 농장 평가 | [v10_farm_infer](v10_farm_infer/REPORT.md): 흑점병 재현율 약 46%, 정밀도 약 63% (1인 판정 추정치) |
| 최신 평가 | [ESP 카메라 교체](v10_farm_infer/esp_cam_20261103/RESULTS.md): 6장, 잎 2,976개, 이상 판정 5개 (정답 라벨 없음) |
| 다음 작업 | 상단 가지치기 후 3시간 간격 추론 확인, 실제 흑점 잎으로 2단계 재학습, 실제 구멍 잎 확보, Jetson 측정 |

## 폴더 지도
| 폴더 | 유형 | 내용 | 리포트 |
|---|---|---|---|
| `common/` | 공용 | 원본 잎 데이터, 실제 사진, 공용 도구 | [DATASETS.md](DATASETS.md) |
| `phase1_cls/` | 기록 | 초기 분류 노트북 (실행 안 됨) | - |
| `v1/` | 데이터 + 모델 | 자동 생성 배경 합성 | [REPORT](v1/REPORT.md) |
| `v2/` | 데이터 + 모델 | 실제 배경 합성 | [REPORT](v2/REPORT.md) |
| `v3_twostage/` | 실험 | 2단계 구조 검증 | [REPORT](v3_twostage/REPORT.md) |
| `v4_reallabel/` | 데이터 + 모델 | 실제 잎 누끼 추가 | [REPORT](v4_reallabel/REPORT.md) |
| `v5_cube/` | 평가 | 큐브 사진, 화이트밸런스 시험 | [REPORT](v5_cube/REPORT.md) |
| `v6_holefilter/` | 코드 | 색 보정 + Hole 필터 | [REPORT](v6_holefilter/REPORT.md) |
| `v7_leafid/` | 구조 + 모델 | 2단계 구조 | [REPORT](v7_leafid/REPORT.md) |
| `v8_reallabel/` | 데이터 + 모델 | 실제 사진 라벨로 재학습 | [REPORT](v8_reallabel/REPORT.md) |
| `v9_fastinfer/` | 코드 | 추론 속도 개선 | [REPORT](v9_fastinfer/REPORT.md) |
| `v10_farm_infer/` | 평가 | 실제 농장 사진 평가, ESP 카메라 교체 평가 | [REPORT](v10_farm_infer/REPORT.md) |

가중치 목록은 [MODELS.md](MODELS.md), 데이터 목록은 [DATASETS.md](DATASETS.md)에 있습니다.

## 빠른 실행
```
cd /helpai-storage/model_git
python v9_fastinfer/deploy/smoke_test.py
python v9_fastinfer/deploy/run.py <사진 또는 폴더> --out <출력 폴더>
```

## 버전 관리 규칙
- **태그:** `v1`~`v10`은 각 버전 폴더를 추가한 시점입니다. 재구성한 이력이라 버전 내부의 세부 수정 과정은 없습니다.
- **`main`:** 항상 동작하는 상태를 유지합니다.
- **브랜치:** 작업마다 `feature/`, `fix/`, `exp/`로 만들고, 검증 후 합칩니다.
- **기존 v1~v10:** 재현 기록이므로 코드를 고치지 않습니다. 현재 실행 경로만 예외입니다.
- **새 버전:** 코드만 바뀌면 폴더를 복사하지 않고 브랜치에서 고칩니다. 새 실험과 평가는 결과 폴더를 새로 만듭니다.

## 공용 도구 (common/tools)
| 스크립트 | 용도 |
|---|---|
| `predict_real.py` | 실사 사진 예측 |
| `compare_real.py` | 여러 모델 실사 비교 |
| `eval_test.py` | 데이터셋 평가 |
| `ckpt_info.py` | 가중치의 에포크·점수 확인 |
| `viz_labels.py` | 합성 데이터 라벨 시각화 |
| `crop_class.py` | 클래스별 크롭 |

## 주의
- **opencv 충돌:** 서버에서는 `import paths` 또는 `import leafdx`를 먼저 해야 `ultralytics`를 불러올 수 있습니다.
- **옛 경로:** `runs/*/args.yaml`과 보관 스크립트에는 당시 경로가 남아 있습니다. 기록용입니다.
- **제외 파일:** 압축 파일, YOLO 캐시, `common/weights`, `weights/`는 Git에서 제외했습니다. 사전학습 가중치는 [MODELS.md](MODELS.md)를 보고 다시 받습니다.
- **push:** 1회 2GB 제한이 있어 태그별로 나눠 올립니다.
