#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Build the indexes of the filtered-search example (src/cagra_filtered_indexes_example.cu) for every
# .fbin file of a directory, one index a file, as many at once as there are GPUs, then search each
# one at a 1% and a 10% filter.
#
#   build_filtered_indexes.sh DATA_DIR OUT_DIR [EXAMPLE_BINARY]
#
# DATA_DIR/NAME.fbin (uint32 rows, uint32 dimension, float32 rows) becomes OUT_DIR/NAME/ with
# graph.cagra (FP16 rows and graph: the two-hop walk), rows.f16 and codes.fp8 (the graph-free index
# of the FP8 filtered scan; the codes also let a graph index choose the scan per search). A build of
# 46.5 million 65-dimension rows takes about two minutes on a B200 and needs about 30 GB of device
# memory. NGPUS overrides the number of GPUs used; a file whose OUT_DIR/NAME/codes.fp8 exists is
# skipped, so the script can be run again after a failure.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '7,17p' "$0"
  exit 2
fi
DATA_DIR=$1
OUT_DIR=$2
EXAMPLE=${3:-$(dirname "$0")/../build/CAGRA_FILTERED_INDEXES_EXAMPLE}
NGPUS=${NGPUS:-$(nvidia-smi --list-gpus | wc -l)}

if [[ ! -x "$EXAMPLE" ]]; then
  echo "no example binary at $EXAMPLE: build the examples first (examples/build.sh)" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

one() {
  local fbin=$1 gpu=$2 name
  name=$(basename "$fbin" .fbin)
  [[ -f "$OUT_DIR/$name/codes.fp8" ]] && return 0
  mkdir -p "$OUT_DIR/$name"
  if CUDA_VISIBLE_DEVICES=$gpu "$EXAMPLE" build "$fbin" "$OUT_DIR/$name" > "$OUT_DIR/$name/build.log" 2>&1; then
    echo "$name: built on GPU $gpu"
  else
    echo "$name: FAILED, see $OUT_DIR/$name/build.log"
    rm -f "$OUT_DIR/$name/codes.fp8"
  fi
}

i=0
for fbin in "$DATA_DIR"/*.fbin; do
  one "$fbin" $((i % NGPUS)) &
  i=$((i + 1))
  (( i % NGPUS == 0 )) && wait
done
wait

for dir in "$OUT_DIR"/*/; do
  [[ -f "$dir/codes.fp8" ]] || continue
  echo "== $(basename "$dir")"
  "$EXAMPLE" search "$dir" | tee "$dir/search.log"
done
