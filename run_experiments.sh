#!/bin/bash
# run_experiments.sh
# This script runs experiments for each swap strategy (1–4) and each swap interval (10, 5, 2, 1).

# Fixed parameters
TKN_DIM=128
QK_DIM=64
NHEADS=8
HN_MULT=4.0
ATTN_BETA=0.125
TIME_STEPS=12
BLOCKS=6
EPOCHS=300
BATCH_SIZE=128
LR=5e-5
DATA_PATH="./data"
NUM_PROCESSES=1

# Loop over swap strategies (1, 2, 3, 4)
for SWAP_STRATEGY in 2 3 4; do
  # Loop over swap intervals (10, 5, 2, 1)
  for SWAP_INTERVAL in 10 5 2 1; do
    echo "--------------------------------------------------"
    echo "Running experiment with swap_strategy=${SWAP_STRATEGY} and swap_interval=${SWAP_INTERVAL}"
    echo "--------------------------------------------------"

      accelerate launch \
      --num-processes ${NUM_PROCESSES} \
      --mixed_precision bf16 \
      --dynamo_backend inductor \
      train.py \
      --tkn-dim ${TKN_DIM} \
      --qk-dim ${QK_DIM} \
      --nheads ${NHEADS} \
      --hn-mult ${HN_MULT} \
      --attn-beta ${ATTN_BETA} \
      --time-steps ${TIME_STEPS} \
      --blocks ${BLOCKS} \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --lr ${LR} \
      --swap-interval ${SWAP_INTERVAL} \
      --swap-strategy ${SWAP_STRATEGY} \
      --data-path ${DATA_PATH}
  done
done
