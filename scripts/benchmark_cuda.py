import torch
import time
import sys
import os

sys.path.append(os.getcwd())
from src.model.llm_wrapper import LLMWrapper
from src.inference.stream_decoder import StreamGraphGenerator
from src.model.probes import LinearProbe

def benchmark():
    print("Loading Model for Benchmark...")
    # Update with your local path
    model_path = r"C:\Users\valno\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659"
    
    llm = LLMWrapper(model_path, device="cuda")
    llm.model.eval()
    
    # Dummy inputs
    dummy_input = torch.randint(0, 1000, (1, 10)).cuda()

    # 1. Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = llm.model(dummy_input)

    # 2. Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Benchmarking Forward Pass Latency...")
    latencies = []
    
    # Benchmark plain forward pass (no hook) to set baseline? 
    # Or benchmark WITH hook? 
    # User said "Average Forward Pass Latency".
    # Usually we want to know the added overhead.
    # Let's measure with hook.
    
    # Register dummy probes
    ent_probe = LinearProbe(4096, 3).half().cuda()
    rel_probe = LinearProbe(4096, 10).half().cuda()
    generator = StreamGraphGenerator(llm.model, ent_probe, rel_probe)
    # The generator attaches hook on init? No, usually explicitly or during init.
    # checking stream_decoder code... it registers in __init__.
    
    # We benchmark the underlying model call which now triggers the hook
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            start_event.record()
            
            # Forward pass
            # This triggers the hook in StreamGraphGenerator
            _ = llm.model(dummy_input)
            
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    # Report
    avg = sum(latencies) / len(latencies)
    print(f"Average Forward Pass Latency (with Hook): {avg:.3f} ms")
    
    # Remove hooks and benchmark baseline
    generator.hook.remove()
    latencies_baseline = []
    print("Benchmarking Pure Baseline...")
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            start_event.record()
            _ = llm.model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            latencies_baseline.append(start_event.elapsed_time(end_event))
            
    avg_base = sum(latencies_baseline) / len(latencies_baseline)
    print(f"Average Baseline Latency: {avg_base:.3f} ms")
    print(f"StreamGraph Overhead: {avg - avg_base:.3f} ms")

if __name__ == "__main__":
    benchmark()
