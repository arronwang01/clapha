#!/bin/bash
# Startup script for an unattended il.train run on one GCP GPU VM (tools/gcp/launch.sh).
# Pulls code, the fl:il checkpoint and the converted replays from the bucket, trains, copies
# logs every minute and checkpoints every five to the bucket, writes out/DONE, and deletes this
# VM. The VM is also created with a hard max-run-duration, so a hang cannot keep billing.
# Needs the VM's service account to read and write the bucket (roles/storage.objectAdmin on
# it); the project's default compute account has no roles here unless granted.
# Instance metadata: clapha-bucket, clapha-mode (smoke | full), clapha-args (il.train flags).
# the log goes to the serial console too: readable with get-serial-port-output even when the VM
# cannot reach the bucket
exec > >(tee -a /var/log/clapha-train.log > /dev/ttyS0) 2>&1
set -x
meta() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
B=$(meta attributes/clapha-bucket)
MODE=$(meta attributes/clapha-mode)
ARGS=$(meta attributes/clapha-args)
NAME=$(meta name)
ZONE=$(meta zone | awk -F/ '{print $NF}')
OUT="$B/out/$NAME"
W=/opt/clapha-train
mkdir -p "$W" && cd "$W"

finish() {
  gcloud storage cp /var/log/clapha-train.log "$OUT/startup.log" -q
  [ -d "$W/clapha/runs/train" ] && gcloud storage rsync -r "$W/clapha/runs/train" "$OUT/train" -q
  echo "$1 $(date -u +%FT%TZ)" | gcloud storage cp - "$OUT/DONE" -q
  echo "CLAPHA-TRAIN-DONE $1"
  # the launching Mac deletes the VM when it sees DONE (tools/gcp/watch.sh); this is
  # the fallback when the VM's own account may delete it; max-run-duration is the last resort
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true
}

( while true; do
    gcloud storage cp /var/log/clapha-train.log "$OUT/startup.log" -q
    [ -f "$W/clapha/runs/train.log" ] && gcloud storage cp "$W/clapha/runs/train.log" "$OUT/train.log" -q
    sleep 60
  done ) &
( while true; do
    sleep 300
    [ -d "$W/clapha/runs/train" ] && gcloud storage rsync -r "$W/clapha/runs/train" "$OUT/train" -q
  done ) &

for _ in $(seq 1 60); do nvidia-smi && break; sleep 10; done
nvidia-smi || { finish "no-gpu"; exit 1; }

# a Python 3.11+ that already has CUDA torch (the image's); our extra packages in a venv on top
PY=""
for candidate in /opt/conda/bin/python /usr/bin/python3 $(ls /opt/*/bin/python 2>/dev/null); do
  "$candidate" -c "import sys, torch; assert sys.version_info >= (3, 11) and torch.cuda.is_available()" && { PY=$candidate; break; }
done
[ -n "$PY" ] || { finish "no-python-with-cuda-torch"; exit 1; }
"$PY" -m venv --system-site-packages /opt/clapha-venv || { finish "venv"; exit 1; }
/opt/clapha-venv/bin/pip install -q orjson zstandard safetensors || { finish "pip"; exit 1; }
PY=/opt/clapha-venv/bin/python

gcloud storage cp "$B/clapha-train-code.zip" . -q && python3 -m zipfile -e clapha-train-code.zip . || { finish "code"; exit 1; }
mkdir -p clapha/ref-firstlight/checkpoints/IL clapha/runs/conv-hog26
gcloud storage cp "$B/ckpt/IL/checkpoint-step-00029396.pt" clapha/ref-firstlight/checkpoints/IL/ -q
gcloud storage rsync -r "$B/conv-hog26" clapha/runs/conv-hog26 -q || { finish "data"; exit 1; }
cd clapha
if [ "$MODE" = smoke ]; then
  "$PY" -m il.train --frames runs/conv-hog26 --out runs/train --smoke $ARGS > runs/train.log 2>&1
else
  "$PY" -m il.train --frames runs/conv-hog26 --out runs/train $ARGS > runs/train.log 2>&1
fi
status=$?
gcloud storage cp runs/train.log "$OUT/train.log" -q
finish "exit-$status"
