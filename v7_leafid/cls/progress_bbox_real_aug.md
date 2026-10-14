# v7 2단계 분류 모델 비교 학습 진행 상황 (bbox_real_aug)
마지막 갱신: 2026-10-14 04:01:36 (UTC)

| 단계 | 상태 | 시작 | 갱신 | 최근 메시지 |
|---|---|---|---|---|
| effnetv2s | 완료 | 03:51:38 | 04:01:34 | v7 real crops (603): {'Black': 7, 'Fresh': 587, 'Hole': 5, 'Yellow': 4} |
| compare | 완료 | 04:01:36 | 04:01:36 | 1 models compared -> compare_bbox_real_aug.md |
| synth_anom | 완료 | 03:50:40 | 03:50:59 | done: test_real_anom/Black 67, test_real_anom/Hole 67, test_real_anom/Yellow 67, train/Black 312, train/Hole 313, train/Yellow 313, val/Black 85, val/Hole 84, val/Yellow 84 (check: real/anom_check.jpg) |

## effnetv2s
```
03:51:38 start
03:52:11 epoch 1/20 train_loss 1.0757 val_acc 0.8734 val_macroF1 0.842 (best 0.842)
03:52:41 epoch 2/20 train_loss 0.5787 val_acc 0.898 val_macroF1 0.8824 (best 0.8824)
03:53:11 epoch 3/20 train_loss 0.4958 val_acc 0.9054 val_macroF1 0.8907 (best 0.8907)
03:53:39 epoch 4/20 train_loss 0.4603 val_acc 0.9005 val_macroF1 0.8892 (best 0.8907)
03:54:08 epoch 5/20 train_loss 0.4215 val_acc 0.9038 val_macroF1 0.8873 (best 0.8907)
03:54:38 epoch 6/20 train_loss 0.4051 val_acc 0.9095 val_macroF1 0.8935 (best 0.8935)
03:55:07 epoch 7/20 train_loss 0.4068 val_acc 0.9169 val_macroF1 0.9045 (best 0.9045)
03:55:36 epoch 8/20 train_loss 0.3839 val_acc 0.9021 val_macroF1 0.8835 (best 0.9045)
03:56:05 epoch 9/20 train_loss 0.3809 val_acc 0.8988 val_macroF1 0.8848 (best 0.9045)
03:56:35 epoch 10/20 train_loss 0.3693 val_acc 0.9005 val_macroF1 0.8775 (best 0.9045)
03:57:06 epoch 11/20 train_loss 0.3695 val_acc 0.9186 val_macroF1 0.906 (best 0.906)
03:57:35 epoch 12/20 train_loss 0.3623 val_acc 0.9178 val_macroF1 0.9003 (best 0.906)
03:58:05 epoch 13/20 train_loss 0.3596 val_acc 0.9252 val_macroF1 0.9145 (best 0.9145)
03:58:34 epoch 14/20 train_loss 0.3559 val_acc 0.9211 val_macroF1 0.9081 (best 0.9145)
03:59:04 epoch 15/20 train_loss 0.3543 val_acc 0.9285 val_macroF1 0.9158 (best 0.9158)
03:59:33 epoch 16/20 train_loss 0.3549 val_acc 0.9227 val_macroF1 0.9115 (best 0.9158)
04:00:03 epoch 17/20 train_loss 0.3536 val_acc 0.9293 val_macroF1 0.9168 (best 0.9168)
04:00:33 epoch 18/20 train_loss 0.3523 val_acc 0.9252 val_macroF1 0.9131 (best 0.9168)
04:01:02 epoch 19/20 train_loss 0.3518 val_acc 0.926 val_macroF1 0.9138 (best 0.9168)
04:01:31 epoch 20/20 train_loss 0.3524 val_acc 0.9219 val_macroF1 0.9099 (best 0.9168)
04:01:31 training done in 9.9 min, evaluating best
04:01:32 test: acc 0.9282 macroF1 0.8927 recall {'Black': 0.863, 'Fresh': 0.976, 'Hole': 0.881, 'Yellow': 0.714}
04:01:33 test_deg: acc 0.8867 macroF1 0.8616 recall {'Black': 0.825, 'Fresh': 0.928, 'Hole': 0.836, 'Yellow': 0.857}
04:01:34 v7 real crops (603): {'Black': 7, 'Fresh': 587, 'Hole': 5, 'Yellow': 4}
```

## compare
```
04:01:36 start
04:01:36 1 models compared -> compare_bbox_real_aug.md
```

## synth_anom
```
03:50:40 start
03:50:41 cube_05.jpg: 52 leaves
03:50:42 cube_06.jpg: 42 leaves
03:50:43 cube_07.jpg: 57 leaves
03:50:44 cube_08.jpg: 56 leaves
03:50:45 cube_09.jpg: 61 leaves
03:50:46 cube_10.jpg: 30 leaves
03:50:47 cube_11.jpg: 101 leaves
03:50:48 cube_12.jpg: 64 leaves
03:50:48 cube_13.jpg: 67 leaves
03:50:49 cube_14.jpg: 70 leaves
03:50:50 cube_15.jpg: 50 leaves
03:50:51 cube_16.jpg: 84 leaves
03:50:52 cube_17.jpg: 152 leaves
03:50:53 cube_18.jpg: 45 leaves
03:50:54 cube_19.jpg: 85 leaves
03:50:55 cube_20.jpg: 39 leaves
03:50:56 cube_21.jpg: 63 leaves
03:50:57 cube_22.jpg: 138 leaves
03:50:58 rose_real.webp: 74 leaves
03:50:59 rose_real2.webp: 62 leaves
03:50:59 done: test_real_anom/Black 67, test_real_anom/Hole 67, test_real_anom/Yellow 67, train/Black 312, train/Hole 313, train/Yellow 313, val/Black 85, val/Hole 84, val/Yellow 84 (check: real/anom_check.jpg)
```

## 비교 결과 (bbox_real_aug)

| 모델 | test acc / macro-F1 | 변형 test acc / macro-F1 | 변형 test 재현율 (Black/Fresh/Hole/Yellow) | ms/장 | 실사 이상판정 비율 (전체) | 실사 이상판정 비율 (판정 대상만) | 판정 대상 분포 (Black/Fresh/Hole/Yellow) |
|---|---|---|---|---|---|---|---|
| effnetv2s | 0.9282 / 0.8927 | 0.8867 / 0.8616 | 0.825/0.928/0.836/0.857 | 1.86 | 0.027 (603) | 0.009 (462) | 2/458/1/1 |