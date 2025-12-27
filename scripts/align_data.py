import torch
import os
import hydra
import sys
sys.path.append(os.getcwd())
from omegaconf import DictConfig
from tqdm import tqdm
from src.model.llm_wrapper import LLMWrapper
from src.dataset import TokenAlignedDataset
from torch.utils.data import DataLoader

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("Pre-aligning Dataset...")
    
    # We need the tokenizer
    # We can load just tokenizer to be fast, but LLMWrapper does it
    # Let's trust LLMWrapper for consistency
    # Use "cpu" to save VRAM for now?
    device = "cpu"
    print(f"Loading Tokenizer from {cfg.model.name_or_path}...")
    llm = LLMWrapper(cfg.model.name_or_path, device=device)
    
    data_root = "data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en"
    save_dir = "data/processed/aligned"
    os.makedirs(save_dir, exist_ok=True)
    
    splits = ["train", "dev", "test"]
    
    for split in splits:
        print(f"\nProcessing {split}...")
        dataset = TokenAlignedDataset(data_root, llm.tokenizer, max_len=cfg.dataset.max_len, split=split)
        
        # We iterate and save
        # Since TokenAlignedDataset.__getitem__ does the work, we just loop data loader
        # batch_size 1 is fine, we just want to collect
        # Actually batch_size can be larger to use parallelism if num_workers > 0?
        # Tokenizer parallelism issues on Windows... stick to num_workers=0, batch=1
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        all_data = []
        for batch in tqdm(loader):
            # batch contains tensors with batch_dim=1.
            # We want to store a list of dicts or a collated tensor object
            # Storing tensors is better for cache_features
            
            # Squeeze
            item = {
                'input_ids': batch['input_ids'].squeeze(0),
                'attention_mask': batch['attention_mask'].squeeze(0),
                'entity_labels': batch['entity_labels'].squeeze(0),
                'rel_labels': batch['rel_labels'].squeeze(0)
            }
            all_data.append(item)
            
        print(f"Saving {len(all_data)} items to {save_dir}/{split}_aligned.pt")
        torch.save(all_data, f"{save_dir}/{split}_aligned.pt")

if __name__ == "__main__":
    main()
