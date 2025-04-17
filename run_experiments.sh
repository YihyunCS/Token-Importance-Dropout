#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p plots

# Set experiment names
BASELINE="baseline"
EXPERIMENT="token_importance_dropout"

echo "===== Starting Baseline Experiment ====="
# Turn off token dropout for baseline
sed -i 's/USE_TOKEN_DROPOUT = True/USE_TOKEN_DROPOUT = False/' src/config.py
python src/train.py $BASELINE

echo "===== Starting Token Importance Dropout Experiment ====="
# Turn on token dropout for experiment
sed -i 's/USE_TOKEN_DROPOUT = False/USE_TOKEN_DROPOUT = True/' src/config.py
# Set dropout method to random or your preferred method
sed -i "s/method='random'/method='random'/" src/token_dropout.py
python src/train.py $EXPERIMENT

echo "===== Generating Comparison Plots ====="
python src/plot_results.py $BASELINE $EXPERIMENT

echo "===== All Experiments Completed ====="
echo "Results are available in logs/ and plots/ directories" 