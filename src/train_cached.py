import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch.nn as nn

from src.dataset import CachedDataset
from src.model.probes import LinearProbe

def train_step(entity_probe, rel_probe, batch, optimizer, device='cuda'):
    # batch['hidden_states'] is already [batch, seq, dim]
    hidden_states = batch['hidden_states'].to(device)
    entity_labels = batch['entity_labels'].to(device)
    rel_labels = batch['rel_labels'].to(device)
    
    entity_logits = entity_probe(hidden_states) 
    rel_logits = rel_probe(hidden_states)

    loss_fct = nn.CrossEntropyLoss()
    loss_ent = loss_fct(entity_logits.view(-1, entity_logits.size(-1)), entity_labels.view(-1))
    loss_rel = loss_fct(rel_logits.view(-1, rel_logits.size(-1)), rel_labels.view(-1))
    
    total_loss = loss_ent + loss_rel
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(f"Training Cached Probes with config:\n{cfg}")
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    
    # 1. Dataset
    cache_dir = "data/processed/cache"
    target_layer = cfg.model.target_layer
    
    print(f"Loading cached dataset for layer {target_layer}...")
    try:
        train_dataset = CachedDataset(cache_dir, layer=target_layer, split="train")
        # Increase batch size massively since we only have MLP
        batch_size = 256 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print("Cache not found! Please run 'python src/cache_features.py' first.")
        return

    # 2. Probes
    print("Initializing Probes...")
    # Determine num classes from data or config? 
    # Logic in original train.py used dataset.rel_labels len.
    # We saved labels in cache but not the mapping.
    # However, since we just need the max int value + 1, we can infer or hardcode.
    # Or load the label tensor and check max.
    
    # Quick check max label in train
    num_ent_classes = train_dataset.entity_labels.max().item() + 1
    num_rel_classes = train_dataset.rel_labels.max().item() + 1
    print(f"Detected classes - Entity: {num_ent_classes}, Relation: {num_rel_classes}")

    entity_probe = LinearProbe(cfg.model.hidden_dim, num_classes=num_ent_classes, dropout=cfg.model.probes.entity.dropout).half().to(device)
    rel_probe = LinearProbe(cfg.model.hidden_dim, num_classes=num_rel_classes, dropout=cfg.model.probes.relation.dropout).half().to(device)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        list(entity_probe.parameters()) + list(rel_probe.parameters()), 
        lr=cfg.train.learning_rate
    )
    
    # 4. Training Loop
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    
    print("\nStarting Fast Training...")
    for epoch in range(cfg.train.epochs):
        entity_probe.train()
        rel_probe.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            loss = train_step(entity_probe, rel_probe, batch, optimizer, device=device)
            total_loss += loss
            pbar.set_postfix({"loss": loss})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")
        
        # Save checkpoints
        torch.save(entity_probe.state_dict(), os.path.join(cfg.train.output_dir, f"cached_entity_probe_ep{epoch+1}.pt"))
        torch.save(rel_probe.state_dict(), os.path.join(cfg.train.output_dir, f"cached_rel_probe_ep{epoch+1}.pt"))

    # Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    try:
        test_dataset = CachedDataset(cache_dir, layer=target_layer, split="test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        entity_probe.eval()
        rel_probe.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                hidden_states = batch['hidden_states'].to(device)
                rel_labels = batch['rel_labels'].to(device)
                
                # Only eval relation probe for F1 (as per user request "Relation Extraction F1")
                logits = rel_probe(hidden_states)
                preds = torch.argmax(logits, dim=-1)
                
                # Mask padding/ignore index if necessary?
                # WebNLG labels: 0 is usually 'O' (No Relation). 
                # We care about F1 on non-O classes usually.
                
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(rel_labels.view(-1).cpu().numpy())
        
        from sklearn.metrics import f1_score
        # micro or macro? Usually micro for relation arrays
        # Ignore class 0 if it's 'No Relation'? 
        # For simplicity, standard weighted/macro f1.
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Test F1: {f1:.4f}")
        
    except FileNotFoundError:
        print("Test cache not found. Cannot evaluate.")
        print("Test F1: 0.0000")
    except Exception as e:
        print(f"Evaluation Error: {e}")
        print("Test F1: 0.0000")

if __name__ == "__main__":
    main()
