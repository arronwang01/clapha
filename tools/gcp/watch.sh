#!/bin/bash
# Wait for a training VM to finish (out/NAME/DONE in the bucket, or CLAPHA-TRAIN-DONE on its serial
# console), copy its results to runs/gcp/NAME, and delete the VM from here.
#   tools/gcp/watch.sh NAME ZONE
set -u
GC=~/google-cloud-sdk/bin/gcloud
B=gs://clapha-train-aa479a94
NAME=$1; ZONE=$2
cd "$(dirname "$0")/../.."
until $GC storage cat "$B/out/$NAME/DONE" 2>/dev/null \
      || $GC compute instances get-serial-port-output "$NAME" --zone "$ZONE" 2>/dev/null | grep -q CLAPHA-TRAIN-DONE \
      || ! $GC compute instances describe "$NAME" --zone "$ZONE" >/dev/null 2>&1; do
  sleep 120
done
mkdir -p "runs/gcp/$NAME"
$GC compute instances get-serial-port-output "$NAME" --zone "$ZONE" > "runs/gcp/$NAME/serial.log" 2>/dev/null || true
$GC storage rsync -r "$B/out/$NAME" "runs/gcp/$NAME" 2>/dev/null || true
$GC compute instances delete "$NAME" --zone "$ZONE" --quiet 2>/dev/null || true
echo "finished $NAME $(date)"
