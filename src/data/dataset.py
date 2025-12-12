import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class AutoRegressiveDataset(Dataset):
    def __init__(self, clean_data_path: str, tokenizer_path: str, max_length: int = 512):
        self.max_length = max_length

        with open(clean_data_path) as f:
            self.lines = [lines.strip() for lines in f if lines.strip()]

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.BOS_id = self.tokenizer.token_to_id("[BOS]")
        self.EOS_id = self.tokenizer.token_to_id("[EOS]")
        self.PAD_id = self.tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self , idx):
        
        text = self.lines[idx]

        id = self.tokenizer.encode(text).ids

        id = [self.BOS_id] + id + [self.EOS_id]

        if len(id) > self.max_length:
            id = id[:self.max_length]

        input_ids = id

        target_ids = id[1:] + [-100]

        attention_mask = [1] * len( input_ids )

        return {

            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask
        }
    

def collate_fn(batch):

    input_id = [data["input_ids"] for data in batch]
    target_id = [data["target_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]

    max_len = max(len(x) for x in input_id)

    padded_input_id = []
    padded_target_id = []   
    padded_attention_mask = []

    padding_id = 0

    for inp , trg , attn in zip(input_id , target_id , attention_mask):
        
        pad_len = max_len - len(inp)

        padded_input_id.append(inp + [padding_id] * pad_len)
        padded_target_id.append(trg + [-100] * pad_len)
        padded_attention_mask.append(attn + [0]*pad_len)


    return {
        "input_ids": torch.tensor(padded_input_id , dtype=torch.long),
        "target_ids": torch.tensor(padded_target_id , dtype=torch.long),  
        "attention_mask": torch.tensor(padded_attention_mask , dtype=torch.long)
    }

