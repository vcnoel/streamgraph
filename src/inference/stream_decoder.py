import torch

class StreamGraphGenerator:
    """
    Zero-Overhead Graph Generator.
    Hooks into the LLM's forward pass to probe hidden states during generation.
    """
    def __init__(self, llm, entity_probe, rel_probe, target_layer=28):
        self.llm = llm
        self.ent_probe = entity_probe
        self.rel_probe = rel_probe
        self.graph_events = []
        self.target_layer = target_layer
        
        # Determine where to hook based on model architecture
        # Defaulting to model.layers for Llama-like models
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
             self.hook_handle = self.llm.model.layers[self.target_layer].register_forward_hook(self._hook_fn)
        else:
            print(f"Warning: Could not automatically find layer {self.target_layer} to hook.")

    def _hook_fn(self, module, input, output):
        # output[0] is the hidden state for the current token being generated
        # Shape: [batch_size, seq_len, hidden_dim]
        # interaction depends on whether we are in prefill or decoding phase
        # But broadly we want the last token's hidden state
        
        if isinstance(output, tuple):
             hidden_state = output[0]
        else:
             hidden_state = output

        # Detach to ensure no gradients leak here during inference
        last_token_hidden = hidden_state[:, -1, :] 
        
        # Run probes immediately (Tensor operations are fast)
        # Using no_grad just in case, though inference mode usually handles it
        with torch.no_grad():
            ent_pred = self.ent_probe(last_token_hidden).argmax(dim=-1).item()
            rel_pred = self.rel_probe(last_token_hidden).argmax(dim=-1).item()
        
        # Store for the graph builder
        self.graph_events.append({
            "token_step": len(self.graph_events),
            "entity": ent_pred,
            "relation": rel_pred
        })

    def generate(self, prompt, **kwargs):
        self.graph_events = [] # Reset events
        # Normal generation. The hook runs in the background.
        return self.llm.generate(prompt, **kwargs)

    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
