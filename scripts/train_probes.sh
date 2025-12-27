#!/bin/bash
# Train the probes
# Usage: ./scripts/train_probes.sh

conda activate gsp
python src/train.py train.batch_size=4 train.epochs=1
