# model/ 디렉토리 구조 (2026-10-13 재구성)

장미 잎 검출·분할(YOLO-seg) 작업 폴더입니다. 버전별로 데이터·학습 결과·평가·스크립트를 나눴습니다.
모든 명령은 `/helpai-storage/model`에서 실행하는 것을 기준으로 적었습니다.

```
model/
├── common/                      # 버전 공용
│   ├── paths.py                 # 공용 경로 + opencv 충돌 우회 (모든 스크립트·노트북이 먼저 import)
│   ├── Dataset/, Dataset.rar    # 원본 누끼 잎 (클래스별 RGBA png)
│   ├── weights/                 # 기본 가중치: yolo26s-seg.pt, yolo11s-seg.pt, yolo26n.pt(AMP 점검용)
│   ├── real_images/             # 실제 재배대 사진 3장 (rose_real3 = 학습에 안 쓴 평가용)
│   ├── tools/                   # 버전 공용 도구 (아래 표)
│   └── reorg/                   # 이번 재구성 스크립트와 이동 로그
├── phase1_cls/
│   └── 01_phase1_cls_baseline.ipynb   # 잎 상태 분류 노트북 (아직 실행 안 됨, 분류용 가중치 업로드 필요)
├── v1/                          # 1차: 자동 생성 배경 합성 데이터
│   ├── 02_phase2_seg_synthetic.ipynb  # v1 데이터 생성 + 학습 노트북 (작업 디렉토리 = v1/)
│   ├── data/rose_seg/           # v1 합성 데이터 (6000/800/800)
│   ├── runs/yolo26s-seg/        # ★ v1 모델 (best = 49 에포크)
│   ├── runs/yolo11s-seg/        # v1 비교군 (32 에포크에서 중단)
│   ├── eval/                    # 테스트 평가(test, test_conf025), 실사 예측(real_pred_conf025/010), best_snapshot.pt, logs/
│   └── scripts/archive/         # 당시 실행한 원본 평가 스크립트 (기록용)
├── v2/                          # 2차: 실제 배경 합성 데이터
    ├── scripts/                 # make_backgrounds → inspect_leaves → make_synth_v2 → train_eval_v2 (+ make_review)
    │   └── archive/             # 공용 도구로 대체된 원본 스크립트 (기록용)
    ├── backgrounds/             # 실사 사진 1,2번에서 추출한 배경 79장
    ├── data/rose_seg_v2/        # v2 합성 데이터 (6000/800/800), rose_seg_v2_smoke/ (시험용 32장)
    ├── runs/ft_from_v1/         # ★ v1에서 이어서 학습 (best = 30 에포크)
    ├── runs/coco_init/          # 기본 가중치에서 학습 (21 에포크에서 중단, best = 20 에포크)
    ├── eval/                    # 실사 예측(ft_from_v1_conf025/010), 비교 이미지(compare_v1_vs_ft.jpg, compare_3models/)
    ├── debug/                   # 배경 추출·잎 샘플·합성 라벨 확인 이미지
    └── logs/                    # 생성·학습·평가 로그
├── v3_twostage/                 # 2단계 구조(잎 영역 찾기 → 이상 상태 분리) 타당성 검증 (README.md 참고)
├── v4_reallabel/                # 실제 사진 잎 라벨링: 모델 초안 + Claude 검토, 실제 잎 누끼 27장, 라벨 초안 (README.md 참고)
├── v5_cube/                     # 큐브 실사 사진 18장 테스트, 화이트밸런스 시험
├── v6_holefilter/               # 추론 파이프라인: v4 + 화이트밸런스 + Hole 후처리 필터 (README.md 참고)
└── v7_leafid/                    # 새 구조 1단계: 잎 탐색 + ID 부여 → (2단계) 잎별 분류 (README.md 참고)
```

## 공용 도구 (common/tools)
| 스크립트 | 용도 | 사용법 |
|---|---|---|
| `predict_real.py` | 실사 사진 예측, 잎마다 마스크+박스+라벨 표시 | `python common/tools/predict_real.py <가중치> <출력폴더> [conf ...]` |
| `compare_real.py` | 여러 모델 실사 비교 (마스크+테두리, 좌상단 범례, Fresh 숨김 버전 포함) | `python common/tools/compare_real.py <출력폴더> 이름=<가중치> ...` |
| `eval_test.py` | 데이터셋 평가 (mAP, 클래스별 AP, 혼동 행렬) | `python common/tools/eval_test.py <가중치> <data.yaml> <출력폴더> [conf] [split]` |
| `ckpt_info.py` | 가중치 파일의 에포크·검증 점수 확인 | `python common/tools/ckpt_info.py <가중치> ...` |
| `viz_labels.py` | 합성 데이터 라벨 시각화 | `python common/tools/viz_labels.py <데이터폴더> <split> <장수> <출력.jpg>` |

## v2 파이프라인 (v2/scripts)
1. `make_backgrounds.py`: 실사 사진에서 잎 없는 배경 추출 → v2/backgrounds, v2/debug (재실행 시 배경을 새로 만듦)
2. `inspect_leaves.py`: 원본 잎 샘플·통계 → v2/debug/leaf_samples.jpg
3. `make_synth_v2.py`: 합성 데이터 생성 → v2/data/rose_seg_v2 (재실행 시 기존 데이터 삭제 후 재생성, `SMOKE=1`이면 시험용 32장)
4. `train_eval_v2.py`: 학습(ft_from_v1, coco_init) + 평가 → v2/runs, v2/eval/final. **이미 best.pt가 있는 실행은 건너뜀** (`--retrain`일 때만 다시 학습)

## 주의
- **opencv 충돌:** 서버 opencv 충돌 때문에 새 파이썬 프로세스에서는 `import paths`를 먼저 해야 `ultralytics`를 불러올 수 있습니다.
- **pip 설치 셀:** 노트북의 `%pip install -U ultralytics` 셀은 충돌의 원인이라 주석 처리했습니다.
- **이전 경로 기록:** `runs/*/args.yaml`에는 학습 당시의 이전 경로(`runs/rose_seg`, `v2_realbg` 등)가 기록되어 있습니다. 기록용이며 사용에는 영향이 없습니다.
- **coco_init 이어서 학습:** `coco_init`을 이어서 학습(resume)하려면 체크포인트 안의 이전 데이터 경로를 고쳐야 합니다.
