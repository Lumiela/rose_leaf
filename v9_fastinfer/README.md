# v9_fastinfer: v8 추론 속도 개선 (출력은 v8과 동일)

**목적 (2026-10-30):** v8 배포 패키지(`v8_reallabel/deploy`)의 사진 1장 추론 시간을 줄입니다. 모델 가중치·설정·판정 기준은 v8과 같고, 출력은 v8과 비트 단위로 같습니다. **v8 폴더는 수정하지 않았습니다.**

## 폴더 구성
```
v9_fastinfer/
├── README.md
├── deploy/                        # ★ 이식용 패키지 (v8 deploy 전체 복사 후 leafdx 수정, 이 폴더만으로 동작)
│   ├── models/                    # v8과 같은 파일 (md5 동일): leaf_seg_v8r2.pt, leaf_cls_v8.pt, states.json
│   ├── leafdx/                    # 수정: stage1.py, stage2.py, pipeline.py, __init__.py (draw.py, _compat.py는 v8과 동일)
│   └── run.py, smoke_test.py, config.json, requirements.txt, samples/, README.md
├── scripts/
│   ├── dump_run.py                # 패키지 1개를 별도 프로세스로 실행 → 시간 + 잎별 요약(마스크 md5 포함) JSON
│   └── compare_v8_v9.py           # v8·v9 비교 (출력 동일 여부 + 속도) → eval/
├── eval/                          # compare_v8_v9.json (사진별 비교·시간), bench.csv (사진별 시간 표)
├── logs/                          # compare_v8_v9.log, cpu_check.log
└── leafdx_deploy_v9_20261030.zip  # deploy/ 압축본 (전송용)
```

## 원인 (v8 단계별 측정, 12MP 사진)
| 병목 | v8 시간 | 원인 |
|---|---|---|
| 1단계 마스크 처리 | L=102 사진 2.5초, L=165 사진 8.3초 | `retina_masks=True`라 잎마다 타일 전체 크기(8L×8L) float 마스크가 생김 → 전부 CPU로 복사(느린 사진 약 7GB)한 뒤 `np.nonzero`로 순회. 비용이 L²에 비례 |
| 2단계 gray world 화이트밸런스 | 약 2초 | 12MP 사진 전체를 float64로 계산 |

- GPU 모델 추론(YOLO + EfficientNet)은 사진당 0.7~1.0초뿐이었습니다 (측정 중 GPU 사용률 평균 9%).
- 같은 사진(20261019_150529)에 L을 강제로 바꾸면 1단계가 L=100: 4.2초 → 130: 6.6초 → 165: 9.2초. 사진 내용이 아니라 L(타일 크기)이 원인입니다.

## 변경 내용
| 파일 | 변경 |
|---|---|
| `leafdx/stage1.py` | 타일별 마스크를 GPU에서 이진화 → 픽셀 수·bbox도 GPU에서 계산 → 잎 bbox 부분만 CPU로 복사. 마스크 크기가 타일과 다를 때는 v8 방식 그대로 처리 (`_tile_masks_cpu`) |
| `leafdx/stage2.py` | `gray_world`: 기준 픽셀 조건(채도 < 60/255, 60 < 밝기 < 240)을 같은 뜻의 정수 비교로 바꿔 GPU에서 계산하고, 채널 이득은 256칸 LUT로 적용. `apply_gain` 추가 |
| `leafdx/pipeline.py` | 간이 화이트밸런스 적용을 LUT로 (평균 계산은 v8과 같음), `gray_world`에 분류기 장치 전달 |
| `leafdx/__init__.py`, `smoke_test.py` | 버전 표기 `v9.2026-10-30`, 설명 |

사용법(API·명령줄·config.json·출력 JSON)은 v8과 같습니다. `from leafdx import LeafPipeline` 그대로 쓰면 됩니다.

## 검증 (`scripts/compare_v8_v9.py`, 서버 V100, 사진 36장 = inputs 27 + inputs_20261019 9, 잎 9,956개)
- **출력 동일:** 36장 모두 잎 수·L·상태 개수가 같고, 잎마다 box·polygon·면적·conf·solidity·선명도·어두운 비율·상태·확률·마스크(md5) 불일치 0건 (`eval/compare_v8_v9.json` → `all_identical: true`)
- `deploy/smoke_test.py` PASS (GPU). CPU 모드(`--device cpu`)도 샘플 1장에서 v8과 동일 (`logs/cpu_check.log`)
- 따라서 v8 정확도(1단계 재현율 0.797 / 정밀도 0.768, 건강 잎 이상 오판 0/865, 합성 이상 검출 99.8%)가 그대로 유지됩니다.

## 속도 (사진 1장, 워밍업 2회 후 2회 평균, 모델 로딩·사진 읽기 제외)
| 해상도 | 사진 | v8 | v9 | 배속 |
|---|---|---|---|---|
| **4000×3000 (12MP, 현재 화분 사진)** | 9 | 12.33초 (1단계 7.74 + 2단계 4.58) | **3.49초** (1.24 + 2.24) | **×3.5** |
| 2560×1920 | 4 | 3.85초 (2.01 + 1.84) | 1.59초 (0.62 + 0.97) | ×2.4 |
| 2048×1536 | 3 | 2.71초 (1.35 + 1.36) | 1.27초 (0.50 + 0.77) | ×2.1 |
| 2000×1500 | 18 | 2.50초 (1.25 + 1.25) | 1.35초 (0.65 + 0.70) | ×1.9 |
| 1280×720 | 2 | 1.08초 (0.47 + 0.60) | 0.79초 (0.31 + 0.47) | ×1.4 |

- 12MP에서 1단계가 더 이상 L에 따라 늘지 않습니다 (v9 12MP 사진 3.06~3.89초).
- **CPU 모드는 빨라지지 않습니다:** 2560×1920 샘플 v8 37.6초 / v9 37.1초. CPU에서는 마스크 처리가 원래 CPU라 이득이 없고, 시간 대부분이 2단계 EfficientNet CPU 추론입니다.

## 남은 시간 (12MP 약 3.5초, 프로파일 값)
- 1단계 약 1.2초: 대부분 YOLO 추론 + 잎 크기 추정
- 2단계 약 2.2초: 크롭 + LANCZOS 리사이즈 0.3~0.7, 잎별 특징(윤곽·선명도·어두운 비율) 0.2~0.55, EfficientNet 0.4~0.5, 간이 화이트밸런스 약 0.37

## 재현 (model/에서 실행)
```
python v9_fastinfer/deploy/smoke_test.py                 # SMOKE_TEST PASS
python v9_fastinfer/scripts/compare_v8_v9.py --reps 2    # 약 13분, eval/ 덮어씀
```

## 주의
- 측정은 서버 V100 기준이며, GPU를 다른 프로세스(메모리 약 20GB)와 함께 쓰는 상태였습니다. **Jetson은 미측정**입니다. 바뀐 부분은 GPU 텐서 연산이라 Jetson GPU에서도 효과가 있을 것으로 예상하지만 확인이 필요합니다.
- 1단계를 TensorRT `.engine`으로 바꿔도 `r.masks.data`가 GPU 텐서이므로 같은 경로로 동작할 것으로 예상합니다 (미확인). 변환 후에는 `smoke_test.py`로 확인하세요.
- 모델 인덱스 `model/README.md`에는 반영하지 않았습니다 (v8도 아직 없음).
