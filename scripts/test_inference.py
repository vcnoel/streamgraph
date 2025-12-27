import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

sys.path.append(os.path.abspath("src"))
from inference.stream_decoder import StreamGraphGenerator
from model.probes import LinearProbe

def test_inference():
    print("Loading Model/Tokenizer for Inference Test...")
    # Update with your local path or use the one we found
    model_path = r"C:\Users\valno\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda")
    
    print("Loading Trained Probes...")
    ent_probe = LinearProbe(4096, 3).half().cuda()
    rel_probe = LinearProbe(4096, 581).half().cuda() # 581 classes as seen in training
    
    # Load checkpoints
    # Assuming epoch 1 for now
    try:
        ent_probe.load_state_dict(torch.load("checkpoints/cached_entity_probe_ep1.pt"))
        rel_probe.load_state_dict(torch.load("checkpoints/cached_rel_probe_ep1.pt"))
        print("Checkpoints loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load checkpoints ({e}). Using random weights.")
    
    print("Initializing StreamGraphGenerator...")
    generator = StreamGraphGenerator(model, ent_probe, rel_probe)
    
    prompt = "Microsoft acquired Activision because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating...")
    # We use the generator.generate wrapper (which calls model.generate but hooks are active)
    # The 'StreamGraphGenerator' definition I wrote:
    # def generate(self, prompt, **kwargs): ...
    # Wait, my implementation of `generate` took `prompt` string.
    # LLM.generate takes inputs.
    # checking my stream_decoder.py...
    # def generate(self, prompt, **kwargs):
    #    return self.llm.generate(prompt, **kwargs)
    # Be careful: 'llm' in StreamGraphGenerator was passed as the model object.
    # If I passed pure model, model.generate expects input_ids usually, or keyword args.
    # If I pass a wrapper, depends on wrapper.
    # In my `probes.py`/`stream_decoder.py` I assumed `llm` is `AutoModelForCausalLM` or similar?
    # No, in `trainer.py` I used `LLMWrapper`.
    # `stream_decoder.py` logic:
    # hook on `self.llm.model.layers`.
    # If `self.llm` is the HF model, then `self.llm.model` might key error if it's not wrapped?
    # LlamaForCausalLM has `.model` attribute (LlamaModel). So `model.model.layers` works.
    
    # My `StreamGraphGenerator.generate` calls `self.llm.generate`.
    # Standard HF generate takes `input_ids`.
    
    outputs = generator.generate(inputs['input_ids'], max_new_tokens=20)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {text}")
    print(f"Graph Events: {len(generator.graph_events)}")
    print(generator.graph_events)

if __name__ == "__main__":
    test_inference()
