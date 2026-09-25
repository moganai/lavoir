#!/usr/bin/env bash
# Full Lavoir recipe on one node (the released model used 16 GPUs with batch 4 each: global batch 64).
#   decision phase -> VOI targets v1 -> joint (2 epochs) -> VOI targets v2 -> joint (1 epoch)
#
#   DATA="data/*.jsonl" OUT=runs/lavoir NGPU=8 BATCH=8 bash scripts/train_from_scratch.sh
#
# DATA must contain the VOI examples (with `slots`, `probes` and `gold`) and may contain any single-turn data.
# Keep BATCH x NGPU (x nodes) at 64 to match the released recipe.
set -euo pipefail
DATA=${DATA:?set DATA to your training JSONL files, e.g. DATA="data/*.jsonl"}
OUT=${OUT:-runs/lavoir}
NGPU=${NGPU:-1}
BATCH=${BATCH:-64}
ENCODER=${ENCODER:-answerdotai/ModernBERT-large}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"

run() { torchrun --standalone --nproc_per_node="$NGPU" -m lavoir.train "$@"; }
targets() {  # $1 = checkpoint, $2 = output file
  python -m lavoir.targets --checkpoint "$1" --data $DATA --out "$2"
}

run --phase decision --encoder "$ENCODER" --data $DATA --out "$OUT/decision" --batch_size "$BATCH" --epochs 4 $EXTRA
targets "$OUT/decision" "$OUT/voi_targets_v1.jsonl"
run --phase joint --init "$OUT/decision" --voi_targets "$OUT/voi_targets_v1.jsonl" --data $DATA \
    --out "$OUT/joint_v1" --batch_size "$BATCH" --epochs 2 $EXTRA
targets "$OUT/joint_v1" "$OUT/voi_targets_v2.jsonl"
run --phase joint --init "$OUT/joint_v1" --voi_targets "$OUT/voi_targets_v2.jsonl" --data $DATA \
    --out "$OUT/joint_v2" --batch_size "$BATCH" --epochs 1 $EXTRA
echo "done: $OUT/joint_v2"
