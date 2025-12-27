import torch
import time
import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from src.model.llm_wrapper import LLMWrapper
from src.dataset import TokenAlignedDataset

class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (128,), dtype=torch.long),
            'entity_labels': torch.zeros(128),
            'rel_labels': torch.zeros(128)
        }

def measure_inference(llm, batch_size=8):
    print(f"Benchmarking Pure Inference (Batch {batch_size})...")
    ds = DummyDataset(100)
    dl = DataLoader(ds, batch_size=batch_size)
    
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            input_ids = batch['input_ids'].cuda()
            _ = llm.model(input_ids, output_hidden_states=True)
            if i >= 5: break
    end = time.time()
    print(f"Inference Speed: {(end-start)/5:.4f} s/batch")

def measure_data_loading(data_root, tokenizer, batch_size=8):
    print(f"Benchmarking Data Loading (Batch {batch_size})...")
    ds = TokenAlignedDataset(data_root, tokenizer, split="dev")
    dl = DataLoader(ds, batch_size=batch_size)
    
    start = time.time()
    for i, batch in enumerate(dl):
        _ = batch['input_ids']
        if i >= 50: break
    end = time.time()
    print(f"Data Loading Speed: {(end-start)/50:.4f} s/batch")

if __name__ == "__main__":
    # Load Model once
    model_path = r"C:\Users\valno\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659"
    try:
        llm = LLMWrapper(model_path, device="cuda")
    except:
        print("Could not load LLM")
        llm = None
        
    if llm:
        measure_inference(llm, batch_size=32)
        measure_inference(llm, batch_size=8)
        
        data_root = "data/raw/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en"
        measure_data_loading(data_root, llm.tokenizer, batch_size=32)
