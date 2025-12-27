# StreamGraph

StreamGraph is a framework for "Zero-Overhead" Knowledge Graph extraction by probing the hidden states of Large Language Models (LLMs) during generation. This approach eliminates the need for autoregressive decoding of structured data (like JSON or triples), significantly reducing inference latency.

## Overview

The core hypothesis is that specific layers of fine-tuned or instructed LLMs (like Llama-3) encode structured semantic information (entities and relations) linearly in their hidden states. By training lightweight linear probes on these states, StreamGraph can extract a Knowledge Graph in real-time as the model generates natural language.

## Project Structure

```
streamgraph/
├── configs/            # Hydra configuration files
├── data/               # Raw and processed datasets (excluded from git)
├── scripts/            # Utility scripts (training, inference, benchmarks)
├── src/
│   ├── dataset.py      # Token alignment and caching logic
│   ├── trainer.py      # Probe training loop
│   ├── cache_features.py # Activation caching optimization
│   ├── inference/      # StreamGraph generation and hooking logic
│   └── model/          # Probe architectures and LLM wrappers
└── requirements.txt    # Architecture dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vcnoel/streamgraph.git
   cd streamgraph
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   conda create -n streamgraph python=3.10
   conda activate streamgraph
   pip install -r requirements.txt
   ```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. 
- **Training Config**: `configs/train.yaml`
- **Model Config**: `configs/model/default.yaml`

Update `configs/model/default.yaml` with the local path to your LLM (e.g., Llama-3-8B-Instruct).

```yaml
model:
  name_or_path: "/path/to/your/llama-3-8b"
  hidden_dim: 4096
  target_layer: 28
```

## Usage

### 1. Data Preparation
StreamGraph requires token-aligned training data. The system currently supports the WebNLG dataset.
Download the raw data into `data/raw/webnlg`.

### 2. Data Alignment (Optimization)
To avoid CPU bottlenecks during feature extraction, pre-align the dataset using the tokenizer.

```bash
python scripts/align_data.py
```
*Output: `data/processed/aligned/*.pt`*

### 3. Feature Caching
Extract and cache LLM hidden states to disk. This loads the pre-aligned data and runs highly efficient GPU inference.

```bash
python -m src.cache_features
```
*Note: This will save tensors to `data/processed/cache`.*

### 4. Training
Train the linear probes using the cached features. This process is highly efficient (minutes).

```bash
python -m src.train_cached
```

Checkpoints will be saved to `src/checkpoints/` (or the configured output dir).

### 4. Inference
Run the inference script to demonstrate the "Zero-Overhead" extraction. This loads the LLM, attaches the probes as forward hooks, and extracts the graph during text generation.

```bash
python scripts/test_inference.py
```

## Benchmarks
(Coming Soon)
Scripts in `scripts/benchmark.sh` will evaluate the latency overhead of the probing mechanism compared to standard generation.

## License
MIT
