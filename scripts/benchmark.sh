#!/bin/bash
# Run latency benchmark
# Usage: ./scripts/benchmark.sh

conda activate gsp
python src/inference/benchmark.py
