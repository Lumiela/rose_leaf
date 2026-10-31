# 붙여넣기 시험: 학습에 안 쓴 Hole 누끼를 실제 농장 사진에 붙여 v9 파이프라인으로 검출 (2026-10-31)

- 누끼: common/Dataset, v7 cls split의 test (학습·검증에 안 쓴 것). Hole 67장 + 대조군 Fresh 67장
- 사진: inputs/minirose 9장(건강한 농장 사진). 기존 FRESH 잎 위에 붙임, 크기 = 사진 L × 1.0~1.4, 무작위 회전·뒤집기
- 조건 (붙인 위치는 같음):
  - asis: 누끼 그대로. Dataset 구멍은 대부분 원본 촬영 배경(흰색·검은색)이 구멍 안에 찍혀 있음 (투명 구멍은 13%뿐)
  - transp: Hole 누끼의 구멍 픽셀(흰·회색 또는 검은색)을 투명하게 → 구멍 너머로 농장 배경(다른 잎·흙)이 보임
- 파이프라인: v9_fastinfer/deploy/run.py, 변경 없음. 매칭: 붙인 잎 영역과 검출 마스크 IoU ≥ 0.5
- 스크립트: paste_make.py → run_paste.sh → paste_eval.py (results.json, sheet_*.jpg), missed_sheet.py (missed_asis.jpg)

| 조건 | 붙인 잎 | 1단계 검출 | ANOMALY:Hole | Black Spot | FRESH | UNKNOWN | 놓침 |
|---|---|---|---|---|---|---|---|
| asis Hole | 67 | 91% | **78%** | 4% | 7% | 1% | 9% |
| transp Hole | 67 | 82% | **67%** | 7% | 6% | 1% | 18% |
| Fresh 대조군 | 67 | 72% | 1 (1.5%) | 1 | 69% | 0 | 28% |

- 1단계에서 찾은 잎만 보면 Hole 판정: asis 85%, transp 82%
- 구멍 면적 8% 이상 누끼: asis 89% → transp 53%. 큰 구멍이 배경으로 비치면 1단계가 잎을 놓치거나(5건) 흑점으로 판정(3건)
- 놓침은 대부분 1단계 문제: 어두운 갈색·보라 누끼(Dataset 촬영 조명)를 잎으로 안 잡음 (missed_asis.jpg)

## 한계
- 이 시험은 Hole 재현율만 봅니다. 실제 농장 흑점 사진에서 Hole 판정은 대부분 흑점 오판이었습니다(gt/RESULTS.md) → 정밀도는 별개 문제
- 붙인 누끼는 Dataset 촬영본이라 조명·색이 농장 잎과 다르고 경계가 티가 납니다. 실제 구멍 난 잎 사진으로 확인 필요
