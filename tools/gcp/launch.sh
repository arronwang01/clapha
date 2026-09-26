#!/bin/bash
# One unattended training VM (startup script tools/gcp/train-startup.sh), capped in time.
#   tools/gcp/launch.sh NAME smoke|full MACHINE HOURS "il.train flags"
#   tools/gcp/launch.sh clapha-smoke smoke g2-standard-4 1 "--init fl:il --extras"
# g2 machines carry one NVIDIA L4. --max-run-duration + DELETE: the VM deletes itself at the
# deadline whatever happens; the script also deletes it when training ends. Watch with
#   gcloud storage cat gs://clapha-train-aa479a94/out/NAME/train.log
set -euo pipefail
GC=~/google-cloud-sdk/bin/gcloud
B=gs://clapha-train-aa479a94
NAME=$1; MODE=$2; MACHINE=${3:-g2-standard-32}; HOURS=${4:-4}; ARGS=${5:-}
cd "$(dirname "$0")/../.."
for ZONE in us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b; do
  if $GC compute instances create "$NAME" --zone "$ZONE" --machine-type "$MACHINE" \
      --image-family pytorch-2-9-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --boot-disk-type pd-balanced --maintenance-policy TERMINATE \
      --max-run-duration "${HOURS}h" --instance-termination-action DELETE --scopes cloud-platform \
      --metadata-from-file startup-script=tools/gcp/train-startup.sh \
      --metadata "clapha-bucket=$B,clapha-mode=$MODE,clapha-args=$ARGS" 2>&1; then
    echo "created $NAME in $ZONE"; exit 0
  fi
done
echo "no zone had capacity" >&2; exit 1
