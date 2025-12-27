import sys
import os
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath("src"))

from dataset import TokenAlignedDataset

def test_alignment():
    print("Loading tokenizer...")
    # Use the local model path we found
    model_path = r"C:\Users\valno\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        print("Could not load local tokenizer, falling back to meta-llama/Meta-Llama-3-8B-Instruct (might require net)")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing Dataset...")
    # Point to the raw data dir we verified
    data_root = "data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en"
    
    # We use 'dev' split for testing as it is smaller
    ds = TokenAlignedDataset(data_root, tokenizer, split="dev")
    
    print(f"Dataset size: {len(ds)}")
    
    # Check first few samples
    for i in range(min(5, len(ds))):
        sample = ds[i]
        input_ids = sample['input_ids']
        ent_labels = sample['entity_labels']
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\nSample {i}:")
        # Print only non-padding
        real_len = (input_ids != tokenizer.pad_token_id).sum().item()
        
        print("Text Tokens + Labels:")
        for j in range(real_len):
            t = tokens[j]
            e = ent_labels[j].item()
            if e != 0:
                print(f"{t} [{e}]", end=" ")
            else:
                print(f"{t}", end=" ")
        print("\n")
        
        # Check if we have any entity labels at all
        if ent_labels.sum() == 0:
            print("WARNING: No entities found in this sample!")
            
if __name__ == "__main__":
    test_alignment()
