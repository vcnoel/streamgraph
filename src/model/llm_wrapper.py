from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMWrapper:
    """
    Wraps a generic LLM to handle loading, freezing, and hidden state access.
    """
    def __init__(self, model_name_or_path, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure pad token is set (common issue with Llama)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        
        # Freeze the model immediately
        self.freeze()
        
    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_hidden_states(self, input_ids, layer_idx):
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[layer_idx]
        
    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens)
