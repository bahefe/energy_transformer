#!/bin/bash
# run_experiments.sh
# This script runs experiments for each swap strategy (2–4) and each swap interval (10, 5, 2, 1).

# Fixed parameters
TKN_DIM=128
QK_DIM=64
NHEADS=8
HN_MULT=4.0
ATTN_BETA=0.125
TIME_STEPS=1
BLOCKS=12
EPOCHS=300
BATCH_SIZE=128
LR=5e-5
DATA_PATH="./data"
NUM_PROCESSES=1

# Accelerate parameters
MIXED_PRECISION="bf16"
DYNAMO_BACKEND="inductor"
GRAD_ACCUM="2"
NUM_MACHINES=1

# Set GPU with ID 1
export CUDA_VISIBLE_DEVICES=2

# Loop over swap strategies (1, 2, 3, 4)
for SWAP_STRATEGY in 1, 2, 3, 4; do
  # Loop over swap intervals (10, 5, 2, 1)
  for SWAP_INTERVAL in 10, 5, 2, 1; do
    echo "--------------------------------------------------"
    echo "Running experiment with swap_strategy=${SWAP_STRATEGY} and swap_interval=${SWAP_INTERVAL}"
    echo "--------------------------------------------------"
    
    accelerate launch \
      --num-processes ${NUM_PROCESSES} \
      --num-machines ${NUM_MACHINES} \
      --mixed_precision ${MIXED_PRECISION} \
      --dynamo_backend ${DYNAMO_BACKEND} \
      --gradient_accumulation_steps ${GRAD_ACCUM} \
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
