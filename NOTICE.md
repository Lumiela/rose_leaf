# 데이터와 코드 출처

이 저장소는 비공개입니다. 공개하기 전에 아래 항목을 확인합니다.

## 공개 데이터셋
`common/Dataset`과 이를 사용한 합성 데이터(`v1`, `v2`, `v4_reallabel`, `v7_leafid`, `v8_reallabel/cls` 등)의 잎 이미지 출처입니다.

Hojayfa, S. S., Islam, J., Mojumdar, M. U. (2024). *Disease Detection in Rose Leaves* (Version 1). Mendeley Data. https://doi.org/10.17632/c2hvz3t6t9.1

- 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 수정 내용: 배경 합성, 크기 조정, 합성 이상 추가 등
- 출처 확인: 이 데이터를 받아 사용했음을 확인했습니다(2026-11-02). 클래스별 파일 수(1,379 / 932 / 881 / 174)도 일치합니다.

## 직접 촬영한 사진
`common/real_images*`, `v8_reallabel/inputs_20261019`, `v10_farm_infer/inputs` 등은 직접 촬영한 사진입니다.

- `v10_farm_infer/inputs`의 15장은 GPS 정보를 제거했습니다.
- 공개 전 농장과 시설의 동의, 사진 속 정보, 팀과 과정 규정을 확인합니다.

## 외부 코드와 가중치
| 항목 | 저장소 포함 | 확인할 점 |
|---|---|---|
| Ultralytics YOLO | 패키지는 미포함, 학습 결과 가중치 포함 | AGPL-3.0 또는 상용 라이선스 조건 |
| DINOv2 코드·가중치 | 미포함 | 연구용 라이선스 파일 포함 |
| SAM2.1, EfficientNet 사전학습 가중치 | 미포함 | 각 배포 조건 |

법률 자문이 아니며, 공개 전 각 원문을 확인합니다.
