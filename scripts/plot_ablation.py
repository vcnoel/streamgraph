import matplotlib.pyplot as plt
import subprocess
import re
import sys
import os

# Ensure we're in the right directory or handle paths
# Using subprocess to call train_cached.py module

layers = [20, 24, 28, 32]
f1_scores = []

print("Starting Layer Ablation Experiment...")

for layer in layers:
    print(f"--- Probing Layer {layer} ---")
    
    # Run the training script via subprocess
    # We pass override arguments to hydra
    # Expecting src/train_cached.py to run, train, and print final metrics
    # We need to make sure src/train_cached.py prints "Test F1: ..." 
    # Currently it prints Avg Loss. Let's update it or rely on logs?
    # The user instruction says "Ensure src.train_cached prints 'Test F1: 0.85'".
    # I need to update train_cached.py first to actually evaluate on test set and print this!
    
    command = [
        "python", "-m", "src.train_cached", 
        f"model.target_layer={layer}",
        "train.epochs=3" # Fast training
    ]
    
    # We'll need to capture output
    # Since train_cached runs for a bit, we might want to stream or wait
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Regex parse
        # usage: Test F1: 0.85
        # We need to implement evaluation in train_cached.py
        match = re.search(r"Test F1: (\d+\.\d+)", result.stdout)
        if match:
            f1 = float(match.group(1))
            f1_scores.append(f1)
            print(f"Layer {layer}: F1 = {f1}")
        else:
            print(f"Warning: Could not parse F1 for layer {layer}. Output snippet:\n{result.stdout[-200:]}")
            f1_scores.append(0.0)
            
    except subprocess.CalledProcessError as e:
        print(f"Error training layer {layer}: {e}")
        print(e.stderr)
        f1_scores.append(0.0)

# The Plot
plt.figure(figsize=(8, 5))
plt.plot(layers, f1_scores, marker='o', linestyle='-', color='b')
plt.title("Semantic Solidity: Extraction Accuracy vs. Network Depth")
plt.xlabel("Llama-3 Layer Depth")
plt.ylabel("Relation Extraction F1")
plt.grid(True)
plt.savefig("ablation_plot.png")
print("Plot saved to ablation_plot.png")
