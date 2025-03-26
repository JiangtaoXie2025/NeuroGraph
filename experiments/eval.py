#!/usr/bin/env bash
# This script serves as a one-click entry point for evaluating the trained model.
# You can set additional flags for specifying checkpoint paths, metrics, etc.

# Example usage: ./eval.sh

CHECKPOINT="outputs/checkpoints/latest_model.pth"
DATA_PATH="data/processed"

echo "Evaluating with checkpoint=${CHECKPOINT}, data_path=${DATA_PATH}"

python src/main.py \
    --data-path ${DATA_PATH} \
    --checkpoint-dir $(dirname ${CHECKPOINT}) \
    # Add any additional flags for evaluation logic
    # For instance, you might have a separate function or mode for eval in main.py
