#!/usr/bin/env bash
# Fine-tune a released Lavoir checkpoint on your own data.
#
#   MODE=decision  single-turn labeled data only (no questions): updates the decision head and the encoder,
#                  keeps the VOI head and its calibration temperature.
#   MODE=voi       data with candidate questions (`slots`, `probes`, `gold`): computes VOI targets with the
#                  released model, then runs one joint epoch (decision + VOI). Mix in some single-turn data
#                  to keep general behaviour.
#
#   INIT=path/or/hub-id DATA="my_data/*.jsonl" MODE=voi OUT=runs/ft NGPU=1 bash scripts/finetune.sh
set -euo pipefail
INIT=${INIT:?set INIT to the released checkpoint (directory or Hub repo id)}
DATA=${DATA:?set DATA to your JSONL files}
MODE=${MODE:-voi}
OUT=${OUT:-runs/finetune}
NGPU=${NGPU:-1}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-1}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"

run() { torchrun --standalone --nproc_per_node="$NGPU" -m lavoir.train "$@"; }

if [ "$MODE" = decision ]; then
  run --phase decision --init "$INIT" --data $DATA --out "$OUT/decision" --batch_size "$BATCH" \
      --epochs "$EPOCHS" --lr_encoder 1e-5 $EXTRA
  echo "done: $OUT/decision"
elif [ "$MODE" = voi ]; then
  python -m lavoir.targets --checkpoint "$INIT" --data $DATA --out "$OUT/voi_targets.jsonl"
  run --phase joint --init "$INIT" --voi_targets "$OUT/voi_targets.jsonl" --data $DATA --out "$OUT/joint" \
      --batch_size "$BATCH" --epochs "$EPOCHS" $EXTRA
  echo "done: $OUT/joint"
else
  echo "MODE must be decision or voi" >&2
  exit 1
fi
