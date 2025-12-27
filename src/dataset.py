import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import glob
import os
import re

class TokenAlignedDataset(Dataset):
    """
    Dataset that aligns raw WebNLG triples to tokenizer tokens.
    """
    def __init__(self, data_root, tokenizer, max_len=128, start_token_id=None, end_token_id=None, split="train"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_root = data_root
        self.split = split
        self.samples = self._load_data()
        
        # Mappings for labels (created dynamically or fixed)
        self.entity_labels = {"O": 0, "B-ENT": 1, "I-ENT": 2}
        self.rel_labels = {"O": 0} # Will expand as we see relations
        self._build_rel_vocab()

    def _load_data(self):
        # Find all xml files
        # Data structure: data_root/release_v3.0/en/{split}/**/*.xml
        # But user might point data_root to 'data/raw/webnlg' or deeper
        # We'll search recursively
        xml_files = glob.glob(os.path.join(self.data_root, "**", "*.xml"), recursive=True)
        # Filter by split if possible (assumes 'train', 'dev', 'test' in path)
        xml_files = [f for f in xml_files if self.split in f or f == xml_files[0]] # Fallback to something if split not found
        
        samples = []
        for f in xml_files:
            try:
                tree = ET.parse(f)
                root = tree.getroot()
                for entry in root.findall('.//entry'):
                    mtriples = [t.text for t in entry.findall('.//mtriple')]
                    lexs = [l.text for l in entry.findall('.//lex')]
                    
                    for lex in lexs:
                         samples.append({
                             "text": lex,
                             "triples": mtriples
                         })
            except Exception as e:
                # print(f"Error parsing {f}: {e}")
                continue
        print(f"Loaded {len(samples)} samples from {len(xml_files)} files.")
        return samples

    def _build_rel_vocab(self):
        # Simple pass to find all relations
        relations = set()
        for s in self.samples:
            for t in s['triples']:
                parts = t.split(' | ')
                if len(parts) == 3:
                     relations.add(parts[1].strip())
        
        for i, r in enumerate(sorted(list(relations))):
            self.rel_labels[f"B-REL-{r}"] = len(self.rel_labels)
            self.rel_labels[f"I-REL-{r}"] = len(self.rel_labels)
            
        print(f"Vocabulary: {len(self.rel_labels)} relation labels.")

    def _normalize_entity(self, ent):
        # Remove underscores
        ent = ent.replace('_', ' ')
        # Remove parentheses for disambiguation e.g. " (United States)"
        ent = re.sub(r'\s*\(.*?\)', '', ent)
        return ent.strip()

    def _align(self, text, triples, input_ids):
        # Create label tensors
        seq_len = len(input_ids)
        ent_label_ids = torch.zeros(seq_len, dtype=torch.long)
        rel_label_ids = torch.zeros(seq_len, dtype=torch.long)
        
        full_tokens = input_ids.tolist()
        
        for t in triples:
            parts = t.split(' | ')
            if len(parts) != 3: continue
            
            subj_raw, pred, obj_raw = parts
            subj = self._normalize_entity(subj_raw)
            obj = self._normalize_entity(obj_raw)
            
            # Helper to match variants
            def match_entity(entity_text, label_id_b, label_id_i):
                # Try raw
                t1 = self.tokenizer(entity_text, add_special_tokens=False)['input_ids']
                # Try with space prefix (common in Llama/GPT for middle of sentence)
                t2 = self.tokenizer(" " + entity_text, add_special_tokens=False)['input_ids']
                
                start, end = self._find_subsequence(full_tokens, t1)
                if start == -1:
                    start, end = self._find_subsequence(full_tokens, t2)
                
                if start != -1:
                    ent_label_ids[start] = label_id_b
                    if end > start + 1:
                        ent_label_ids[start+1:end] = label_id_i
                    return True
                return False

            # Match Subject
            match_entity(subj, self.entity_labels["B-ENT"], self.entity_labels["I-ENT"])
            
            # Match Object
            match_entity(obj, self.entity_labels["B-ENT"], self.entity_labels["I-ENT"])

            # Match Relation (Predicate)
            # Try to match the predicate text
            pred_tokens1 = self.tokenizer(pred, add_special_tokens=False)['input_ids']
            pred_tokens2 = self.tokenizer(" " + pred, add_special_tokens=False)['input_ids']
            
            p_start, p_end = self._find_subsequence(full_tokens, pred_tokens1)
            if p_start == -1:
                p_start, p_end = self._find_subsequence(full_tokens, pred_tokens2)
            
            rel_tag_b = self.rel_labels.get(f"B-REL-{pred}", 0)
            rel_tag_i = self.rel_labels.get(f"I-REL-{pred}", 0)
            
            if p_start != -1 and rel_tag_b != 0:
                 rel_label_ids[p_start] = rel_tag_b
                 rel_label_ids[p_start+1:p_end] = rel_tag_i

        return ent_label_ids, rel_label_ids

    def _find_subsequence(self, main, sub):
        n = len(main)
        m = len(sub)
        if m == 0: return -1, -1
        for i in range(n - m + 1):
            if main[i:i+m] == sub:
                return i, i+m
        return -1, -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        triples = item['triples']
        
        # Encoding
        encodings = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        
        ent_labels, rel_labels = self._align(text, triples, input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'entity_labels': ent_labels,
            'rel_labels': rel_labels
        }

class CachedDataset(Dataset):
    """
    Dataset that loads pre-computed hidden states and labels from disk.
    Attributes:
        tensors: [N, seq_len, hidden_dim]
        entity_labels: [N, seq_len]
        rel_labels: [N, seq_len]
    """
    def __init__(self, cache_dir, layer, split="train"):
        print(f"Loading {split} cache for layer {layer}...")
        self.tensors = torch.load(f"{cache_dir}/{split}_layer_{layer}.pt")
        labels = torch.load(f"{cache_dir}/{split}_labels.pt")
        self.entity_labels = labels['entity_labels']
        self.rel_labels = labels['rel_labels']
        
    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.tensors[idx],
            "entity_labels": self.entity_labels[idx],
            "rel_labels": self.rel_labels[idx]
        }
