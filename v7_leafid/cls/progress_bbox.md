# v7 2단계 분류 모델 비교 학습 진행 상황 (bbox)
마지막 갱신: 2026-10-14 03:05:46 (UTC)

| 단계 | 상태 | 시작 | 갱신 | 최근 메시지 |
|---|---|---|---|---|
| export_bbox | 완료 | 02:25:37 | 02:25:39 | done: 603 crops, hold 141 -> /helpai-storage/model/v7_leafid/runs/real_images_camfix_20261014_005116/bbox |
| bg_size | 완료 | 02:25:39 | 02:28:30 | global median 88px -> bg_leaf_size.json |
| render | 완료 | 02:28:30 | 02:30:27 | dataset ready -> /helpai-storage/model/v7_leafid/cls/data_bbox (check: bbox_samples_check.jpg) |
| yolo11s_cls | 완료 | 02:30:29 | 02:37:36 | v7 real crops (603): {'Black': 41, 'Fresh': 321, 'Hole': 158, 'Yellow': 83} |
| effnetv2s | 완료 | 02:37:41 | 02:44:47 | v7 real crops (603): {'Black': 11, 'Fresh': 491, 'Hole': 88, 'Yellow': 13} |
| dinov2_vits14 | 완료 | 02:44:52 | 03:05:44 | v7 real crops (603): {'Black': 15, 'Fresh': 453, 'Hole': 21, 'Yellow': 114} |
| compare | 완료 | 03:05:46 | 03:05:46 | 3 models compared -> compare_bbox.md |

## export_bbox
```
02:25:37 start
02:25:37 camfix_01.jpg: 101 crops, hold 24
02:25:37 camfix_02.jpg: 87 crops, hold 24
02:25:38 camfix_03.jpg: 92 crops, hold 23
02:25:38 camfix_04.jpg: 187 crops, hold 43
02:25:39 camfix_05.jpg: 136 crops, hold 27
02:25:39 done: 603 crops, hold 141 -> /helpai-storage/model/v7_leafid/runs/real_images_camfix_20261014_005116/bbox
```

## bg_size
```
02:25:39 leaf search on background photos
02:27:57 bg_cube: 2394 leaves so far
02:28:30 bg_real: 2874 leaves so far
02:28:30 cube_05.jpg: leaf long side median 104px (n=105)
02:28:30 cube_06.jpg: leaf long side median 86px (n=88)
02:28:30 cube_07.jpg: leaf long side median 99px (n=133)
02:28:30 cube_08.jpg: leaf long side median 93px (n=109)
02:28:30 cube_09.jpg: leaf long side median 77px (n=120)
02:28:30 cube_10.jpg: leaf long side median 84px (n=68)
02:28:30 cube_11.jpg: leaf long side median 79px (n=103)
02:28:30 cube_12.jpg: leaf long side median 79px (n=123)
02:28:30 cube_13.jpg: leaf long side median 89px (n=143)
02:28:30 cube_14.jpg: leaf long side median 82px (n=119)
02:28:30 cube_15.jpg: leaf long side median 102px (n=80)
02:28:30 cube_16.jpg: leaf long side median 85px (n=167)
02:28:30 cube_17.jpg: leaf long side median 89px (n=157)
02:28:30 cube_18.jpg: leaf long side median 78px (n=82)
02:28:30 cube_19.jpg: leaf long side median 92px (n=156)
02:28:30 cube_20.jpg: leaf long side median 85px (n=80)
02:28:30 cube_21.jpg: leaf long side median 84px (n=65)
02:28:30 cube_22.jpg: leaf long side median 82px (n=142)
02:28:30 rose_real.webp: leaf long side median 83px (n=149)
02:28:30 rose_real2.webp: leaf long side median 92px (n=143)
02:28:30 rose_real3.webp: leaf long side median 97px (n=120)
02:28:30 global median 88px -> bg_leaf_size.json
```

## render
```
02:28:30 start (backgrounds: 20 photos, scale-matched)
02:29:04 train 500/2348 originals
02:29:17 train 1000/2348 originals
02:29:28 train 1500/2348 originals
02:29:46 train 2000/2348 originals
02:30:09 train: 7047 images
02:30:18 val: 710 images
02:30:26 test: 724 images
02:30:27 dataset ready -> /helpai-storage/model/v7_leafid/cls/data_bbox (check: bbox_samples_check.jpg)
```

## yolo11s_cls
```
02:33:02 epoch 10/30 train_loss nan val_top1 0.80563
02:33:16 epoch 11/30 train_loss nan val_top1 0.83521
02:33:30 epoch 12/30 train_loss nan val_top1 0.83521
02:33:43 epoch 13/30 train_loss nan val_top1 0.8338
02:33:57 epoch 14/30 train_loss nan val_top1 0.81408
02:34:11 epoch 15/30 train_loss nan val_top1 0.83662
02:34:24 epoch 16/30 train_loss nan val_top1 0.82817
02:34:38 epoch 17/30 train_loss nan val_top1 0.79577
02:34:52 epoch 18/30 train_loss nan val_top1 0.75775
02:35:06 epoch 19/30 train_loss nan val_top1 0.8662
02:35:19 epoch 20/30 train_loss nan val_top1 0.84085
02:35:35 epoch 21/30 train_loss nan val_top1 0.83662
02:35:49 epoch 22/30 train_loss nan val_top1 0.85211
02:36:03 epoch 23/30 train_loss nan val_top1 0.82676
02:36:17 epoch 24/30 train_loss nan val_top1 0.84789
02:36:31 epoch 25/30 train_loss nan val_top1 0.81972
02:36:45 epoch 26/30 train_loss nan val_top1 0.84225
02:36:59 epoch 27/30 train_loss nan val_top1 0.82676
02:37:13 epoch 28/30 train_loss nan val_top1 0.84648
02:37:27 epoch 29/30 train_loss nan val_top1 0.83944
02:37:31 epoch 30/30 train_loss nan val_top1 0.8661971688270569
02:37:32 training done in 7.0 min, evaluating
02:37:35 test: acc 0.8812 macroF1 0.7945 recall {'Black': 0.812, 'Fresh': 0.928, 'Hole': 0.866, 'Yellow': 0.429}
02:37:35 test_deg: acc 0.884 macroF1 0.7916 recall {'Black': 0.812, 'Fresh': 0.928, 'Hole': 0.866, 'Yellow': 0.571}
02:37:36 v7 real crops (603): {'Black': 41, 'Fresh': 321, 'Hole': 158, 'Yellow': 83}
```

## effnetv2s
```
02:37:41 start
02:38:06 epoch 1/20 train_loss 1.1511 val_acc 0.7732 val_macroF1 0.6451 (best 0.6451)
02:38:27 epoch 2/20 train_loss 0.6257 val_acc 0.8254 val_macroF1 0.7137 (best 0.7137)
02:38:48 epoch 3/20 train_loss 0.5319 val_acc 0.8451 val_macroF1 0.7353 (best 0.7353)
02:39:09 epoch 4/20 train_loss 0.4664 val_acc 0.8197 val_macroF1 0.7125 (best 0.7353)
02:39:31 epoch 5/20 train_loss 0.4438 val_acc 0.8634 val_macroF1 0.7431 (best 0.7431)
02:39:51 epoch 6/20 train_loss 0.4107 val_acc 0.8394 val_macroF1 0.7387 (best 0.7431)
02:40:12 epoch 7/20 train_loss 0.3972 val_acc 0.8535 val_macroF1 0.7783 (best 0.7783)
02:40:33 epoch 8/20 train_loss 0.4009 val_acc 0.8718 val_macroF1 0.7737 (best 0.7783)
02:40:55 epoch 9/20 train_loss 0.3818 val_acc 0.862 val_macroF1 0.7585 (best 0.7783)
02:41:16 epoch 10/20 train_loss 0.3808 val_acc 0.862 val_macroF1 0.7659 (best 0.7783)
02:41:36 epoch 11/20 train_loss 0.3674 val_acc 0.8493 val_macroF1 0.7445 (best 0.7783)
02:41:57 epoch 12/20 train_loss 0.3629 val_acc 0.8775 val_macroF1 0.7815 (best 0.7815)
02:42:18 epoch 13/20 train_loss 0.3560 val_acc 0.8775 val_macroF1 0.789 (best 0.789)
02:42:39 epoch 14/20 train_loss 0.3555 val_acc 0.8789 val_macroF1 0.778 (best 0.789)
02:43:00 epoch 15/20 train_loss 0.3537 val_acc 0.869 val_macroF1 0.7659 (best 0.789)
02:43:21 epoch 16/20 train_loss 0.3533 val_acc 0.8704 val_macroF1 0.7614 (best 0.789)
02:43:42 epoch 17/20 train_loss 0.3556 val_acc 0.8746 val_macroF1 0.7736 (best 0.789)
02:44:03 epoch 18/20 train_loss 0.3524 val_acc 0.8761 val_macroF1 0.7722 (best 0.789)
02:44:24 epoch 19/20 train_loss 0.3527 val_acc 0.8789 val_macroF1 0.7692 (best 0.789)
02:44:44 epoch 20/20 train_loss 0.3522 val_acc 0.8704 val_macroF1 0.762 (best 0.789)
02:44:44 training done in 7.0 min, evaluating best
02:44:45 test: acc 0.9199 macroF1 0.8585 recall {'Black': 0.85, 'Fresh': 0.981, 'Hole': 0.851, 'Yellow': 0.571}
02:44:46 test_deg: acc 0.8729 macroF1 0.8247 recall {'Black': 0.787, 'Fresh': 0.918, 'Hole': 0.836, 'Yellow': 0.857}
02:44:47 v7 real crops (603): {'Black': 11, 'Fresh': 491, 'Hole': 88, 'Yellow': 13}
```

## dinov2_vits14
```
02:44:52 start
02:54:13 train features (14094, 768) in 557s
02:55:20 C=0.01 val_macroF1 0.6073
02:56:25 C=0.03 val_macroF1 0.6066
02:59:03 C=0.1 val_macroF1 0.6194
03:02:25 C=0.3 val_macroF1 0.5993
03:05:01 C=1.0 val_macroF1 0.6043
03:05:01 best C=0.1 val_macroF1 0.6194; evaluating
03:05:16 test: acc 0.8011 macroF1 0.6655 recall {'Black': 0.725, 'Fresh': 0.856, 'Hole': 0.761, 'Yellow': 0.429}
03:05:29 test_deg: acc 0.7597 macroF1 0.6406 recall {'Black': 0.812, 'Fresh': 0.75, 'Hole': 0.731, 'Yellow': 0.714}
03:05:44 v7 real crops (603): {'Black': 15, 'Fresh': 453, 'Hole': 21, 'Yellow': 114}
```

## compare
```
03:05:46 start
03:05:46 3 models compared -> compare_bbox.md
```

## 비교 결과 (bbox)

| 모델 | test acc / macro-F1 | 변형 test acc / macro-F1 | 변형 test 재현율 (Black/Fresh/Hole/Yellow) | ms/장 | 실사 이상판정 비율 (전체) | 실사 이상판정 비율 (판정 대상만) | 판정 대상 분포 (Black/Fresh/Hole/Yellow) |
|---|---|---|---|---|---|---|---|
| dinov2_vits14 | 0.8011 / 0.6655 | 0.7597 / 0.6406 | 0.812/0.75/0.731/0.714 | 35.88 | 0.249 (603) | 0.258 (462) | 15/343/14/90 |
| effnetv2s | 0.9199 / 0.8585 | 0.8729 / 0.8247 | 0.787/0.918/0.836/0.857 | 1.76 | 0.186 (603) | 0.162 (462) | 7/387/59/9 |
| yolo11s_cls | 0.8812 / 0.7945 | 0.884 / 0.7916 | 0.812/0.928/0.866/0.571 | 1.58 | 0.468 (603) | 0.403 (462) | 23/276/109/54 |