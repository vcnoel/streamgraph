import torch
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model.llm_wrapper import LLMWrapper
from src.dataset import TokenAlignedDataset

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("Initializing Feature Caching...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Layers to cache (as requested: 20, 24, 28, 32)
    target_layers = [20, 24, 28, 32] 
    
    # 1. Load LLM
    print(f"Loading LLM from {cfg.model.name_or_path}...")
    llm = LLMWrapper(cfg.model.name_or_path, device=device)
    llm.model.eval()
    
    # 2. Dataset
    data_root = "data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en"
    # Create output dir
    save_dir = "data/processed/cache"
    os.makedirs(save_dir, exist_ok=True)

    splits = ["train", "dev"]
    
    for split in splits:
        print(f"\nProcessing split: {split}")
        dataset = TokenAlignedDataset(data_root, llm.tokenizer, max_len=cfg.dataset.max_len, split=split)
        
        # REDUCED BATCH SIZE FOR SAFETY
        extract_batch_size = 8
        dataloader = DataLoader(dataset, batch_size=extract_batch_size, shuffle=False)
        
        layer_buffers = {l: [] for l in target_layers}
        label_ent_buffer = []
        label_rel_buffer = []
        
        chunk_idx = 0
        
        # Chunking strategy
        chunk_size = 100 # Flush every 100 batches
        
        def flush_chunk(idx, buffer, ent_buf, rel_buf):
            print(f"Flushing chunk {idx}...")
            # Save temporary files
            torch.save({
                "ent": torch.cat(ent_buf, dim=0),
                "rel": torch.cat(rel_buf, dim=0)
            }, f"{save_dir}/{split}_chunk_{idx}_labels.pt")
            
            for layer in target_layers:
                if len(buffer[layer]) > 0:
                     torch.save(torch.cat(buffer[layer], dim=0), f"{save_dir}/{split}_chunk_{idx}_layer_{layer}.pt")
            
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_ids = batch['input_ids'].to(device)
                
                # Debug print once
                if i == 0:
                    print(f"Input device: {input_ids.device}")
                    print(f"Model device: {llm.model.device}")
                
                # Forward pass
                outputs = llm.model(input_ids, output_hidden_states=True)
                
                # Extract and Move to CPU
                for layer in target_layers:
                    if layer < len(outputs.hidden_states):
                        h = outputs.hidden_states[layer].half().cpu()
                        layer_buffers[layer].append(h)
                
                label_ent_buffer.append(batch['entity_labels'])
                label_rel_buffer.append(batch['rel_labels'])
                
                if (i + 1) % chunk_size == 0:
                     flush_chunk(chunk_idx, layer_buffers, label_ent_buffer, label_rel_buffer)
                     # Reset
                     layer_buffers = {l: [] for l in target_layers}
                     label_ent_buffer = []
                     label_rel_buffer = []
                     chunk_idx += 1
        
        # Flush remaining
        if len(label_ent_buffer) > 0:
             flush_chunk(chunk_idx, layer_buffers, label_ent_buffer, label_rel_buffer)
             chunk_idx += 1
        
        # Merge all chunks
        print(f"Merging {chunk_idx} chunks for {split}...")
        
        # Merge Labels
        all_ent = []
        all_rel = []
        for c in range(chunk_idx):
             lbls = torch.load(f"{save_dir}/{split}_chunk_{c}_labels.pt")
             all_ent.append(lbls['ent'])
             all_rel.append(lbls['rel'])
             os.remove(f"{save_dir}/{split}_chunk_{c}_labels.pt") # Cleanup
             
        torch.save({
            "entity_labels": torch.cat(all_ent, dim=0), 
            "rel_labels": torch.cat(all_rel, dim=0)
        }, f"{save_dir}/{split}_labels.pt")
        
        # Merge Layers
        for layer in target_layers:
            layer_tensors = []
            for c in range(chunk_idx):
                fpath = f"{save_dir}/{split}_chunk_{c}_layer_{layer}.pt"
                if os.path.exists(fpath):
                    layer_tensors.append(torch.load(fpath))
                    os.remove(fpath) # Cleanup
            
            if len(layer_tensors) > 0:
                full_tensor = torch.cat(layer_tensors, dim=0)
                torch.save(full_tensor, f"{save_dir}/{split}_layer_{layer}.pt")
                print(f"Saved merged Layer {layer}: {full_tensor.shape}")
        
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
