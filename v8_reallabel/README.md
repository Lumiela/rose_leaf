# v8_reallabel: 실제 사진 잎 라벨링 + 잎 탐색(1클래스 YOLO-seg) 학습

**목적 (2026-10-14~15):** 실제 사진의 잎을 소엽 단위로 라벨링하고, 1단계 잎 탐색 모델을 실제 사진으로 학습·평가합니다. 기존 v4/v7은 합성 데이터로만 학습했습니다.

## 폴더 구성
```
v8_reallabel/
├── README.md, scenes.json     # scenes.json = 장면 단위 split (train 20 / val 3 / test 4) + 제외한 중복 사진 목록
├── inputs/                    # 실제 사진 27장 (심볼릭 링크, 중복 제거한 고유 장면만)
├── v7in/, v7boxes/            # v7 잎 탐색 실행 입력/결과 (SAM 초안의 박스 프롬프트로 사용)
├── work/<사진>/               # ★ 라벨 원본: inst.npz (잎 ID 맵), meta.json (제외 영역, reviewed, 수정 기록), edits_*.json, history/ (undo)
├── review/<사진>/             # 검토 이미지: overview.jpg, 확대 타일 t<NN>.jpg, tiles.json, props.json
├── data/leaf_seg/             # 학습용 내보내기: images|labels/{train,val} (타일), full/{val,test} (원본+정답), leaf_seg.yaml, stats.json
├── runs/                      # 학습 결과: r1_fromv4, r1_coco11s (weights/best.pt)
├── eval/                      # test 평가: <모델>/summary.json, <모델>/<사진>.jpg, compare.md
├── logs/                      # 실행 로그
└── scripts/                   # 아래 표
```

## 라벨 기준
- **단위:** 소엽 1장 = 인스턴스 1개. 노란 잎, 병든 잎, 마른 잎도 모두 "잎"입니다.
- **제외 영역(ignore):** 흐려서 경계를 알 수 없는 아래층 잎 무리. 학습할 때는 그 안의 라벨 없는 잎 색 픽셀을 회색으로 칠하고, 평가할 때는 그 안의 예측을 오탐에서 뺍니다.
- **배경:** 줄기, 전선, 화분, 트레이.

## 스크립트 (model/에서 실행)
| 단계 | 스크립트 | 내용 |
|---|---|---|
| 0 | `real_inventory.py` | 서버의 실제 사진 전체 MD5 + 유사 중복 검사 (중복 제거 근거) |
| 1 | `run_drafts.sh` → `sam_draft.py` | inputs/ 구성, v7 잎 탐색, SAM2.1 초안 (v7 박스 + 덮이지 않은 잎 색 씨앗, 가는 물체 제거, 합쳐진 마스크 자동 분할) |
| 2 | `run_grid.sh` | 놓친 잎 보충: 덮이지 않은 잎 색 위에 잎 크기 박스 격자로 SAM 실행 (2회) |
| 3 | `review.py <사진>` | 검토 이미지 생성 (좌표 눈금, ID, 빨강 = 의심, 자홍 = 라벨 없는 잎 색, 회색 = 제외 영역) |
| 4 | `apply_edits.py <사진> <edits.json|undo>` | 수정: del, split, autosplit, add, addpts, addbox, fix, merge, trim, grid, auto, ignore, mark |
| 5 | `export_yolo.py` | 잎 1클래스 YOLO-seg 데이터로 내보내기 (타일 = 잎 길이 L × 8) |
| 6 | `train_v8.py <이름> <초기 가중치>` | 학습 (imgsz 640, 100에폭, patience 30) |
| 7 | `eval_leafseg.py <가중치> <이름>` | test 평가 (원본 해상도 타일 추론, mask IoU ≥ 0.5) / `compare_eval.py` 비교표 |
| 8 | `refine_with_model.py <가중치>` | (예정) 2회차: 모델 예측으로 자동 라벨 보완 (놓친 잎 추가, 합쳐진 마스크 교체). reviewed 사진은 건드리지 않음 |
- 공용: `sam_common.py` (SAM2.1-small = `common/weights/sam2.1_s.pt`), 보조: `auto_eval_r1.sh`, `queue_r1_coco.sh`, `debug_*.py`

## 데이터 현황
| split | 사진 | 라벨 |
|---|---|---|
| test (4) | camfix_02, cube_07, fix_84496bcf, fix_eafda190 | Claude 타일 검토 (reviewed), 사용자 확인 전 |
| val (3) | fix_8b90aa7e, cube_11, cube_17 | 자동 |
| train (20) | camfix_01 (검토), fix 4장, cube 15장 | 자동 (camfix_01만 검토) |

## 결과 (test 4장, 잎 880개)
| 모델 | 재현율 | 정밀도 | F1 | 합쳐진 예측 |
|---|---|---|---|---|
| baseline_v4 (기존 v4, 클래스 무시) | 0.615 | 0.629 | 0.622 | 53 |
| r1_fromv4 23에폭 | 0.785 | 0.783 | 0.784 | 38 |
| r1_fromv4 100에폭 (`runs/r1_fromv4/weights/best.pt`) | 0.801 | 0.761 | 0.780 | 51 |
| r1_coco11s (COCO yolo11s-seg에서 시작, 97에폭) | 0.793 | 0.757 | 0.775 | 54 |
| r2_fromv4 (2회차 라벨, 100에폭, conf 0.25) | 0.831 | 0.719 | 0.771 | 43 |
| **r2_fromv4 conf 0.35 (val로 선택, 현재 기본값)** | **0.797** | **0.768** | **0.782** | **39** |
| r1_fromv4 conf 0.35 | 0.752 | 0.810 | 0.780 | 41 |

### 2회차 (2026-10-15)
- `refine_with_model.py` (r1_fromv4, conf 0.4): train·val 22장에서 놓친 잎 213개 추가, 합쳐진 마스크 152개 교체 → `data/leaf_seg_r2` → `runs/r2_fromv4` (`run_r2.sh`)
- 신뢰도 임계값은 val에서 골랐습니다(`sweep_conf.sh`, 두 모델 모두 0.35). 단, val 라벨도 r1 예측으로 보완됐으므로 완전히 독립적인 선택은 아닙니다.
- **결론:** test F1은 모든 모델이 0.77~0.79로 4장 기준 오차 범위 안입니다. 2회차는 재현율과 정밀도의 균형만 바꾸고 합쳐진 예측을 51 → 39로 줄였습니다. 다음 개선은 모델보다 데이터(고정 카메라 고해상도 사진 추가, test 확대, 정답 확인)가 좌우합니다.

## 주의
- 서버 실제 사진 중 중복: rose_real = cube_22, rose_real2 = cube_19, rose_real3 = cube_07, camfix_04/05 = 고해상도 원본(fix_c6485924/fix_92ccc604)의 축소본. **v7 2단계 평가의 "학습에 안 쓴 사진" cube_22는 학습에 쓴 rose_real과 같은 사진입니다.**
- cube_21은 분홍 조명과 비스듬한 각도 때문에 test에서 train으로 옮겼습니다.

## v7 파이프라인 연결 (2026-10-15)
- **v8 1단계:** `scripts/leaf_search_v8.py` (r2_fromv4, conf 0.35). 잎 크기 L은 축소 사진 1회 예측으로 자동 추정하고, 출력은 v7 형식이라 `v7_leafid/scripts/classify_leaves.py`(2단계)가 그대로 동작합니다.
- **실행:** `scripts/run_pipeline_eval.sh` → `pipeline/v8_stage1` (배경 색 필터 켬), `pipeline/v8_stage1_nobg` (필터 끔), v7 비교용 `v7boxes/stage2`
- **평가:** `scripts/eval_pipeline.py` → `eval/pipeline/summary.json`, 이상 판정 모음 `eval/pipeline/flagged_*_test.jpg`

| test 4장 (잎 880) | 재현율 | 정밀도 | 합쳐진 예측 | 보류율 | 2단계까지 깨끗하게 간 정답 잎 | 판정 대상 중 합쳐짐 | 이상 판정 |
|---|---|---|---|---|---|---|---|
| v7 (v4 + 필터) | 0.595 | 0.851 | 96 | 19.9% | 442 (50.2%) | 62 | 12 (대부분 합쳐진 잎 → Hole 오판) |
| v8 + v7 배경 필터 | 0.610 | 0.767 | 23 | 9.3% | 493 (56.0%) | 20 | 17 |
| **v8, 필터 끔 (권장)** | **0.802** | 0.769 | 31 | **10.2%** | **643 (73.1%)** | **26** | 24 (Yellow 13 = 연한 잎/초기 황변 애매, Black 7·Hole 4 = 대부분 그늘·벽 배경 오판) |

- v7 배경 색 필터는 v4 오탐에 맞춰 만든 것이라 v8에서는 그늘진 진짜 잎을 버립니다 → v8에서는 끕니다.
- 2단계까지 가는 정답 잎이 50% → 73%로 늘었고, 합쳐진 잎이 2단계에 들어가 Hole로 오판되는 문제가 사라졌습니다.
- 남은 문제는 2단계: v8이 넘기는 그늘진·연한 잎에서 Black·Yellow 오판. 2단계는 v7 1단계 크롭으로 학습됐으므로 v8 크롭으로 재학습하면 줄어들 것으로 봅니다.

### 황변 판정 확인 (2026-10-15, 사용자)
- `eval/pipeline/yellow_check.jpg`의 Yellow 판정 13건(v8 필터 끔 + v7 2단계, test 4장)은 **전부 정상**(사용자 확인). 흑점 7·구멍 4건도 그늘·벽 배경 오판이므로, test 판정 대상 865개 중 이상 판정 24건(2.8%)은 모두 오탐입니다.
- 현재 test 4장에는 확인된 실제 이상 잎이 없습니다. → 실제 이상 검출률은 여전히 측정 불가.

## 2단계 재학습: v8 크롭 + 판정 상태 (2026-10-15) — `cls/`
- **데이터** (`cls/scripts/build_cls_v8.py` → `cls/data/`): v7 합성 부분(`v7_leafid/cls/data_bbox`, 하드 링크) + v8 1단계(`pipeline/v8_stage1_nobg`) 판정 대상 잎을 v8 split대로 Fresh로 추가(train 3,954 × 원본·열화, val 639, test 865) + 같은 잎에 합성 이상 1개씩(v7 `synth_real_anom` 그리기 함수). train 18,909 / val 1,988.
- **학습** (`cls/scripts/train_cls_v8.py`): EfficientNetV2-S, 20에폭 (v7과 같은 방식) → `cls/runs/effnetv2s/best.pt`, val macro-F1 0.940
- **판정 상태** (`cls/scripts/calib_states.py`, val에서 선택 → `states.json`): HOLD_SHAPE(합쳐짐·가장자리) / HOLD(품질) / FRESH / ANOMALY:<클래스> / UNKNOWN(이상 의심 → 관찰 대상)
  - val에서 실제 건강 잎 오판 0%, 합성 이상 99.7%라 품질 게이트는 꺼진 값(s_min 0), t_anom 0.5, t_fresh 0.7이 선택됨 (애매한 실제 사례가 생기면 다시 조정)
- **파이프라인 판정** (`cls/scripts/classify_v8.py <run> <사진폴더>` → `<run>/stage2_v8/`)

| test 4장 실제 잎 (865, 사용자 확인 기준 전부 건강) | 이상 판정 | 이상 의심 | 합성 이상 검출 (같은 잎, 865) |
|---|---|---|---|
| v7 2단계 (`runs_bbox_real_aug`) | 24 (2.8%) | — | — |
| **v8 2단계 + 판정 상태** | **0** | **0** | **99.8%** (흑점 100%, 구멍 99.7%, 황변 99.7%) |

- 합성 데이터 test(`data_bbox` test, Dataset 누끼): acc 0.920, 재현율 흑점 0.80 / 구멍 0.85 / 황변 0.86 (v7: 0.928, 0.86 / 0.88 / 0.71) → 실제 사진 잎을 넣어도 이상 민감도가 떨어지지 않음
- 27장 전체: 잎 5,839 → FRESH 5,456, HOLD_SHAPE 381 (6.5%), UNKNOWN 2 (흐린 잎·잎 무리), ANOMALY 0
- **한계:** 합성 이상은 학습과 같은 그리기 방식이라 낙관적. **실제 이상 잎 검출률은 여전히 미측정** (실제 이상 사진 필요). test 4장은 학습 사진과 같은 챔버·카메라라, 새 날짜·새 조명에서의 오판율은 추가 촬영으로 확인 필요.
