import torch
import torch.nn as nn
from tqdm import tqdm

def train_step(llm, entity_probe, rel_probe, batch, optimizer, layer_idx=28, device='cuda'):
    """
    Performs one training step for the probes.
    The LLM is frozen.
    """
    # 1. Freeze LLM (ensure it's in eval mode)
    llm.eval()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    entity_labels = batch['entity_labels'].to(device)
    rel_labels = batch['rel_labels'].to(device)
    
    with torch.no_grad():
        outputs = llm(input_ids, output_hidden_states=True)
        # Extract hidden state from the best layer (e.g., Layer 28 of 32)
        hidden_states = outputs.hidden_states[layer_idx] 
    
    # 2. Train Probes
    # Check if we need to flatten or keep sequence dim depending on label format
    # Assuming labels are [batch, seq_len]
    
    entity_logits = entity_probe(hidden_states) # [batch, seq_len, num_classes]
    rel_logits = rel_probe(hidden_states)       # [batch, seq_len, num_classes]

    # 3. Compute Loss
    # Flatten for CrossEntropy: [batch * seq_len, num_classes]
    loss_fct = nn.CrossEntropyLoss()
    
    loss_ent = loss_fct(entity_logits.view(-1, entity_logits.size(-1)), entity_labels.view(-1))
    loss_rel = loss_fct(rel_logits.view(-1, rel_logits.size(-1)), rel_labels.view(-1))
    
    total_loss = loss_ent + loss_rel
    
    # Optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_ent.item(), loss_rel.item()

def train_epoch(dataloader, llm, entity_probe, rel_probe, optimizer, device='cuda'):
    entity_probe.train()
    rel_probe.train()
    
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        loss, _, _ = train_step(llm, entity_probe, rel_probe, batch, optimizer, device=device)
        total_loss += loss
        
    return total_loss / len(dataloader)
