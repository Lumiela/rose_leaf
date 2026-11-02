# 가중치 목록

SHA-256은 2026-11-02 기준입니다. Git에서 제외한 파일은 원본 `model/`과 `model_git` 폴더에 있습니다.

| 경로 | 크기 | Git | SHA-256 |
|---|---:|---|---|
| `common/weights/dinov2_vits14_pretrain.pth` | 84.2MiB | 제외 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `common/weights/sam2.1_s.pt` | 88.0MiB | 제외 | `60f9e43f1307be192eef341437e02c40f32cd61cf36a97a203a0998a2952873a` |
| `common/weights/torch_hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth` | 82.7MiB | 제외 | `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74` |
| `common/weights/yolo11s-cls.pt` | 13.0MiB | 제외 | `e2b605d1c8c212b434a75a32759a6f7adf1d2b29c35f76bdccd4c794cb653cf2` |
| `common/weights/yolo11s-seg.pt` | 19.7MiB | 제외 | `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3` |
| `common/weights/yolo26n.pt` | 5.3MiB | 제외 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `common/weights/yolo26s-seg.pt` | 22.4MiB | 제외 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |
| `v1/eval/best_snapshot.pt` | 88.1MiB | 포함 | `2c4f73ad617798ba771e1a61a90a4e0e760c30bd1a395c998320be2da15ba44e` |
| `v1/runs/yolo11s-seg/weights/best.pt` | 77.5MiB | 포함 | `fbeb8e477a98e007571212323a6a3f6e93172e3e3c159c93ed31d5885eb66dc2` |
| `v1/runs/yolo11s-seg/weights/last.pt` | 77.5MiB | 포함 | `19530b6e3adc02cb711e3f76f8fd33f8d48408fd63598b45f57b3a185036d8fe` |
| `v1/runs/yolo26s-seg/weights/best.pt` | 22.3MiB | 포함 | `54aaea94e5bef0aa67e7bf58f5f2da0e2bc249272b51209dd85e70ec61013b43` |
| `v1/runs/yolo26s-seg/weights/last.pt` | 22.3MiB | 포함 | `3450373e51b1cc796fd33e0cb08108024e3cf433201f8956ae379bbfce696270` |
| `v2/runs/coco_init/weights/best.pt` | 88.1MiB | 포함 | `c832f944073657d9aa8e1f7d876788ccf8d59509e9b0c301d87102744d0b23e1` |
| `v2/runs/coco_init/weights/last.pt` | 88.1MiB | 포함 | `262064f0ff9bd2a1cb4bf6e5d0f9dc220995bb4ba19f48870431f255998e96c5` |
| `v2/runs/ft_from_v1/weights/best.pt` | 22.3MiB | 포함 | `bb916213a83ff83eccb7f56a6b969e6b0cdd1511925981de023c21de53d8f2fa` |
| `v2/runs/ft_from_v1/weights/last.pt` | 22.3MiB | 포함 | `c0d65b6ad20d334b9dc6aa316c5a25825dcece95f2755c7e9e2353b52907c8e2` |
| `v4_reallabel/runs/ft_from_v2ft/weights/best.pt` | 22.3MiB | 포함 | `050779396b9f276dfa02d274920e355d853374c5739cb95e33444e3b7fb3978c` |
| `v4_reallabel/runs/ft_from_v2ft/weights/last.pt` | 22.3MiB | 포함 | `f1248826bc4a30f46d2c81b3abc94d4259858817792de5c5cd2cc980b94a7283` |
| `v7_leafid/cls/runs/effnetv2s/best.pt` | 77.8MiB | 포함 | `1a1a2ed26521a09826c94269090857cae7b2b58682d89db2712d77b8b4ca9d0b` |
| `v7_leafid/cls/runs/yolo11s_cls/weights/best.pt` | 10.5MiB | 포함 | `dba7511621b2981c9f1c2bc8c9158d58c8c4b98bc53812806b572b7710f9860d` |
| `v7_leafid/cls/runs/yolo11s_cls/weights/last.pt` | 10.5MiB | 포함 | `b893cfc127dc2d3e4da364f17922870f27a7aabe1c7c277e88ccfba1eec6a4b9` |
| `v7_leafid/cls/runs_bbox/effnetv2s/best.pt` | 77.8MiB | 포함 | `5ba9fd0fe7cc5f5caba6011eb35caa7379af8516bc60389ee0ae3cee79d77bbc` |
| `v7_leafid/cls/runs_bbox/yolo11s_cls/weights/best.pt` | 10.5MiB | 포함 | `3485ddb790c54a6ecc2f58fd14a19cb78a63db54aa3652cb011fb6957c9f96cd` |
| `v7_leafid/cls/runs_bbox/yolo11s_cls/weights/last.pt` | 10.5MiB | 포함 | `8ccc8a11f28453a87f9b86e139c1797a67ee27254cda9f451f5d02cd54710436` |
| `v7_leafid/cls/runs_bbox_aborted_022537/effnetv2s/best.pt` | 77.8MiB | 포함 | `edd93384d5b912320666650cfe4148d7ba96c3fc9ab4dd5acf8d90b2f6fb4b41` |
| `v7_leafid/cls/runs_bbox_aborted_022537/yolo11s_cls/weights/best.pt` | 10.5MiB | 포함 | `d9f4b7dbfaf2610334719b2a0b429d5d7c89f4f39ccab67f89d017149cefabcb` |
| `v7_leafid/cls/runs_bbox_aborted_022537/yolo11s_cls/weights/last.pt` | 10.5MiB | 포함 | `4fe59b2af2e626ae89f2b0e616da8cd0d695fc4348ec330be76d8b508924f646` |
| `v7_leafid/cls/runs_bbox_real/effnetv2s/best.pt` | 77.8MiB | 포함 | `9ef8de2256ca414727ce903b66701889c1d6abbe5aa9ec4c3275d3fa8605a1fb` |
| `v7_leafid/cls/runs_bbox_real_aug/effnetv2s/best.pt` | 77.8MiB | 포함 | `53f0cf15b6b552c43f8502cb006e97733252ecf37e396fe557e1f78d2d9a7854` |
| `v8_reallabel/cls/runs/effnetv2s/best.pt` | 77.8MiB | 포함 | `05432fc204821fc1467fec18284cf62ed0cdbdb4a6aff92e97899801c39e7ec0` |
| `v8_reallabel/deploy/models/leaf_cls_v8.pt` | 77.8MiB | 포함 | `05432fc204821fc1467fec18284cf62ed0cdbdb4a6aff92e97899801c39e7ec0` |
| `v8_reallabel/deploy/models/leaf_seg_v8r2.pt` | 22.3MiB | 포함 | `e075ebb4039328ec8a652f2474dc60bace340b3a1340143d53245db18b0034ff` |
| `v8_reallabel/runs/r1_coco11s/weights/best.pt` | 19.6MiB | 포함 | `af5649953786be9a5537f3157f538d687b5e6bcb5369705d3305a4a9241fc077` |
| `v8_reallabel/runs/r1_coco11s/weights/last.pt` | 19.6MiB | 포함 | `b70c14fcd7b9fe06faefaf7fea5e270ceebd49c9a1e86629e33fe28e92e3c32d` |
| `v8_reallabel/runs/r1_fromv4/weights/best.pt` | 22.3MiB | 포함 | `980a9fa50601287faf4c765e1c62388f338158b2f598c43c83677767c21970c4` |
| `v8_reallabel/runs/r1_fromv4/weights/last.pt` | 22.3MiB | 포함 | `9114c37395411095a6e36850a40ab4c3b8fdcab11c91f855ef6bce9a34b96b56` |
| `v8_reallabel/runs/r2_fromv4/weights/best.pt` | 22.3MiB | 포함 | `e075ebb4039328ec8a652f2474dc60bace340b3a1340143d53245db18b0034ff` |
| `v8_reallabel/runs/r2_fromv4/weights/last.pt` | 22.3MiB | 포함 | `b476b2201bbf07daceb7d7c81fe1a28ba18c1510d8454467f4d3304b6968df35` |
| `v9_fastinfer/deploy/models/leaf_cls_v8.pt` | 77.8MiB | 포함 | `05432fc204821fc1467fec18284cf62ed0cdbdb4a6aff92e97899801c39e7ec0` |
| `v9_fastinfer/deploy/models/leaf_seg_v8r2.pt` | 22.3MiB | 포함 | `e075ebb4039328ec8a652f2474dc60bace340b3a1340143d53245db18b0034ff` |
| `weights/yolo26n.pt` | 5.3MiB | 제외 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

## 현재 사용
- 1단계: `v9_fastinfer/deploy/models/leaf_seg_v8r2.pt` (원본 `v8_reallabel/runs/r2_fromv4/weights/best.pt`)
- 2단계: `v9_fastinfer/deploy/models/leaf_cls_v8.pt` (원본 `v8_reallabel/cls/runs/effnetv2s/best.pt`)

같은 해시의 파일은 내용이 같습니다.
