#!/bin/bash
# card.sh <members...>: tile the saved 50-particle matrix to each size and time MemberSchur per call against cuDSS.
# Worker fable-solver's cluster runs; removed before the branch is handed over.
D=$(dirname $0); OUT=${OUT:-/tmp}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in "$@"; do
  python $D/tile_case.py $D/mc50.npz --members $m --out $OUT/t$m.npz
  python $D/percall.py $OUT/t$m.npz --split --cols 1,3 --reps 15 --out $OUT/percall-t$m.json 2>&1 | grep -v Warning
  rm -f $OUT/t$m.npz
done
