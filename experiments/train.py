#!/usr/bin/env bash
# This script serves as a one-click entry point for training the model.
# Adjust flags, hyperparameters, or environment variables as needed.

# Example usage: ./train.sh

# Set default values
DATA_PATH="data/processed"
BATCH_SIZE=8
EPOCHS=10
LR=1e-3

echo "Starting training with data_path=${DATA_PATH}, batch_size=${BATCH_SIZE}, epochs=${EPOCHS}, lr=${LR}"

python src/main.py \
    --data-path ${DATA_PATH} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR}
