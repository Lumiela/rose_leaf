#!/bin/bash
# Download pretrained weights for the classifier comparison into common/weights (no package changes).
cd /helpai-storage/model
P="python -u -c 'import sys; sys.path.insert(0,\"v7_leafid/cls/scripts\"); from cls_common import Progress as P; P.log(\"download\", sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)'"
log() { python -c "import sys; sys.path.insert(0,'v7_leafid/cls/scripts'); from cls_common import Progress as P; P.log('download', sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)" "$@"; }
W=common/weights
mkdir -p $W/torch_hub/checkpoints
log "start"
get() {  # get <url> <dest>
  if [ -s "$2" ]; then log "exists $(basename $2) ($(du -h $2 | cut -f1))"; return 0; fi
  log "downloading $(basename $2) from $1"
  if curl -L --fail --retry 3 -o "$2.part" "$1"; then mv "$2.part" "$2"; log "ok $(basename $2) ($(du -h $2 | cut -f1))"; else rm -f "$2.part"; log "FAILED $(basename $2)"; return 1; fi
}
ok=0
get https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt $W/yolo11s-cls.pt || ok=1
get https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth $W/torch_hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth || ok=1
get https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth $W/dinov2_vits14_pretrain.pth || ok=1
if [ ! -d $W/dinov2_repo ]; then
  get https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip /tmp/dinov2_main.zip && \
  python -c "import zipfile; zipfile.ZipFile('/tmp/dinov2_main.zip').extractall('$W')" && mv $W/dinov2-main $W/dinov2_repo && log "dinov2 code -> $W/dinov2_repo" || ok=1
fi
if [ $ok = 0 ]; then log "all weights ready" done; else log "some downloads failed" failed; fi
