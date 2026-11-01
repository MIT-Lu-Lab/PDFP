#!/bin/bash
# Run the pump on every instance of benchmark/instances.txt with four seeds.
#
#   bash benchmark/run.sh <presolved_directory> <output_directory> [trajectories]
#
# trajectories = 1 (default) or 2 (two concurrent trajectories per run).
# Each run has the whole GPU to itself; set CUDA_VISIBLE_DEVICES to choose it.
set -eu
PRESOLVED=$1
OUT=$2
TRAJ=${3:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
while read -r name; do
    for seed in 1 2 3 4; do
        julia --project="$ROOT" "$ROOT/scripts/solve.jl" --instance "$PRESOLVED/$name.mps" \
            --output_directory "$OUT" --time_limit 1800 --seed "$seed" \
            --trajectories "$TRAJ" --quiet
    done
done < "$ROOT/benchmark/instances.txt"
