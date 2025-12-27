import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

from src.dataset import TokenAlignedDataset
from src.model.probes import LinearProbe
from src.trainer import train_epoch
from src.model.llm_wrapper import LLMWrapper

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(f"Training Probes with config:\n{cfg}")
    
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load LLM (Frozen)
    print("Loading LLM...")
    llm = LLMWrapper(cfg.model.name_or_path, device=device)
    
    # 2. Dataset
    print("Loading Dataset...")
    # Point to the correct data root from config or hardcoded for now based on what we found
    # We found data at: data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en
    data_root = "data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en"
    
    dataset = TokenAlignedDataset(data_root, llm.tokenizer, max_len=cfg.dataset.max_len, split="train")
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    
    # 3. Probes
    print("Initializing Probes...")
    # Entity Probe: 3 classes (O, B-ENT, I-ENT)
    entity_probe = LinearProbe(cfg.model.hidden_dim, num_classes=3, dropout=cfg.model.probes.entity.dropout).half().to(device)
    
    # Relation Probe: Dynamic classes based on dataset
    num_rel_classes = len(dataset.rel_labels)
    print(f"Relation classes: {num_rel_classes}")
    rel_probe = LinearProbe(cfg.model.hidden_dim, num_classes=num_rel_classes, dropout=cfg.model.probes.relation.dropout).half().to(device)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        list(entity_probe.parameters()) + list(rel_probe.parameters()), 
        lr=cfg.train.learning_rate
    )
    
    # 5. Training Loop
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    
    for epoch in range(cfg.train.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.train.epochs}")
        avg_loss = train_epoch(dataloader, llm.model, entity_probe, rel_probe, optimizer, device=device)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        torch.save(entity_probe.state_dict(), os.path.join(cfg.train.output_dir, f"entity_probe_ep{epoch+1}.pt"))
        torch.save(rel_probe.state_dict(), os.path.join(cfg.train.output_dir, f"rel_probe_ep{epoch+1}.pt"))

if __name__ == "__main__":
    main()
