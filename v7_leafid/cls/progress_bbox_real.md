# v7 2단계 분류 모델 비교 학습 진행 상황 (bbox_real)
마지막 갱신: 2026-10-14 03:48:15 (UTC)

| 단계 | 상태 | 시작 | 갱신 | 최근 메시지 |
|---|---|---|---|---|
| effnetv2s | 완료 | 03:39:15 | 03:48:14 | v7 real crops (603): {'Black': 0, 'Fresh': 599, 'Hole': 4, 'Yellow': 0} |
| compare | 완료 | 03:48:15 | 03:48:15 | 1 models compared -> compare_bbox_real.md |
| build_real | 완료 | 03:39:09 | 03:39:13 | synthetic 8487 linked; real healthy leaves: train 938 (x2 images, synthetic Fresh 1930), val 253, test_real 201, excluded U/X 40 |

## effnetv2s
```
03:39:15 start
03:39:46 epoch 1/20 train_loss 1.0747 val_acc 0.8754 val_macroF1 0.7311 (best 0.7311)
03:40:12 epoch 2/20 train_loss 0.5729 val_acc 0.8827 val_macroF1 0.7081 (best 0.7311)
03:40:38 epoch 3/20 train_loss 0.4944 val_acc 0.891 val_macroF1 0.7433 (best 0.7433)
03:41:04 epoch 4/20 train_loss 0.4513 val_acc 0.8972 val_macroF1 0.7437 (best 0.7437)
03:41:30 epoch 5/20 train_loss 0.4149 val_acc 0.8827 val_macroF1 0.7279 (best 0.7437)
03:41:55 epoch 6/20 train_loss 0.4112 val_acc 0.8816 val_macroF1 0.721 (best 0.7437)
03:42:21 epoch 7/20 train_loss 0.3963 val_acc 0.892 val_macroF1 0.7284 (best 0.7437)
03:42:46 epoch 8/20 train_loss 0.3892 val_acc 0.8847 val_macroF1 0.73 (best 0.7437)
03:43:12 epoch 9/20 train_loss 0.3830 val_acc 0.8879 val_macroF1 0.7326 (best 0.7437)
03:43:39 epoch 10/20 train_loss 0.3693 val_acc 0.8982 val_macroF1 0.7453 (best 0.7453)
03:44:06 epoch 11/20 train_loss 0.3652 val_acc 0.892 val_macroF1 0.7398 (best 0.7453)
03:44:33 epoch 12/20 train_loss 0.3648 val_acc 0.9055 val_macroF1 0.7726 (best 0.7726)
03:45:00 epoch 13/20 train_loss 0.3588 val_acc 0.9055 val_macroF1 0.7748 (best 0.7748)
03:45:27 epoch 14/20 train_loss 0.3561 val_acc 0.9065 val_macroF1 0.7734 (best 0.7748)
03:45:55 epoch 15/20 train_loss 0.3539 val_acc 0.9128 val_macroF1 0.7707 (best 0.7748)
03:46:22 epoch 16/20 train_loss 0.3526 val_acc 0.9055 val_macroF1 0.7553 (best 0.7748)
03:46:49 epoch 17/20 train_loss 0.3525 val_acc 0.9076 val_macroF1 0.7618 (best 0.7748)
03:47:17 epoch 18/20 train_loss 0.3526 val_acc 0.9045 val_macroF1 0.7606 (best 0.7748)
03:47:44 epoch 19/20 train_loss 0.3529 val_acc 0.9045 val_macroF1 0.7607 (best 0.7748)
03:48:11 epoch 20/20 train_loss 0.3517 val_acc 0.9065 val_macroF1 0.7656 (best 0.7748)
03:48:11 training done in 8.9 min, evaluating best
03:48:12 test: acc 0.9144 macroF1 0.877 recall {'Black': 0.838, 'Fresh': 0.981, 'Hole': 0.806, 'Yellow': 0.857}
03:48:13 test_deg: acc 0.8923 macroF1 0.849 recall {'Black': 0.85, 'Fresh': 0.942, 'Hole': 0.791, 'Yellow': 0.857}
03:48:14 v7 real crops (603): {'Black': 0, 'Fresh': 599, 'Hole': 4, 'Yellow': 0}
```

## compare
```
03:48:15 start
03:48:15 1 models compared -> compare_bbox_real.md
```

## build_real
```
03:39:09 start
03:39:13 synthetic 8487 linked; real healthy leaves: train 938 (x2 images, synthetic Fresh 1930), val 253, test_real 201, excluded U/X 40
```

## 비교 결과 (bbox_real)

| 모델 | test acc / macro-F1 | 변형 test acc / macro-F1 | 변형 test 재현율 (Black/Fresh/Hole/Yellow) | ms/장 | 실사 이상판정 비율 (전체) | 실사 이상판정 비율 (판정 대상만) | 판정 대상 분포 (Black/Fresh/Hole/Yellow) |
|---|---|---|---|---|---|---|---|
| effnetv2s | 0.9144 / 0.877 | 0.8923 / 0.849 | 0.85/0.942/0.791/0.857 | 1.93 | 0.007 (603) | 0.004 (462) | 0/460/2/0 |