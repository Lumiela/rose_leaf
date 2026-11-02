# 데이터 목록

파일 수와 용량은 2026-11-02 기준입니다. 합성 데이터는 `common/Dataset`과 배경에서 만들었습니다. 출처와 공개 조건은 [NOTICE.md](NOTICE.md)를 보세요.

| 경로 | 파일 | 용량 | 내용 | 출처 |
|---|---:|---:|---|---|
| `common/Dataset` | 3,366 | 1,130MB | 잎 누끼 4클래스 (Fresh 1,379, Black Spot 932, Hole 881, Yellow 174) | 공개 데이터셋 |
| `common/real_images` | 3 | 1MB | 초기 재배대 사진 (`cube_22/19/07`과 중복) | 직접 촬영 |
| `common/real_images_camfix` | 5 | 2MB | 고정 카메라 | 직접 촬영 |
| `common/real_images_cube` | 18 | 8MB | 스마트폰 큐브 사진 | 직접 촬영 |
| `common/real_images_local` | 5 | 2MB | 고정 카메라 고해상도 | 직접 촬영 |
| `v1/data` | 15,204 | 656MB | 합성 장면과 라벨 | 합성 |
| `v2/backgrounds` | 79 | 3MB | 잎 없는 배경 | 직접 촬영에서 추출 |
| `v2/data` | 15,269 | 1,228MB | 실제 배경 합성 | 합성 |
| `v4_reallabel/data` | 15,268 | 1,203MB | 실제 누끼 포함 합성 | 합성 |
| `v4_reallabel/real_cutouts` | 27 | 1MB 미만 | 실제 잎 누끼 | 직접 촬영에서 추출 |
| `v7_leafid/cls/data` | 8,483 | 59MB | 누끼 입력 분류 데이터 | 공개 데이터 렌더링 |
| `v7_leafid/cls/data_bbox` | 8,483 | 93MB | 크롭 입력 분류 데이터 | 합성 |
| `v8_reallabel/data` | 1,152 | 120MB | 실제 사진 타일과 라벨 | 직접 촬영 + 자동·검토 라벨 |
| `v8_reallabel/work` | 254 | 8MB | 라벨 원본 `inst.npz`, 수정 기록 | 라벨 작업 |
| `v8_reallabel/cls/data` | 23,351 | 237MB | 2단계 학습 데이터 | 합성 + 직접 촬영 |
| `v8_reallabel/inputs` | 링크 27 | - | 고유 장면 27장 (`common`의 사진을 가리킴) | 직접 촬영 |
| `v8_reallabel/inputs_20261019` | 9 | 7MB | 12MP 큐브 사진 | 직접 촬영 |
| `v10_farm_infer/inputs` | 22 | 101MB | 농장 사진 (건강 9, 흑점병 13) | 직접 촬영 |
| `v10_farm_infer/inputs_paste` | 18 | 93MB | Hole 누끼 붙여넣기 시험 | 합성 |

## 데이터 분할
- `v1`, `v2`, `v4`: 원본 잎 단위 70/15/15, 장면 6,000 / 800 / 800
- `v8`: 장면 단위 train 20, val 3, test 4 (`v8_reallabel/scenes.json`)

## 제외한 원본
`common/Dataset.rar` 등 압축 파일은 원본 `model/`에만 있습니다.
