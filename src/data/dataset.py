import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class AutoRegressiveDataset(Dataset):
    # Initialize datapath for autoregressive language modeling
    def __init__(self, clean_data_path: str, tokenizer_path: str, max_length: int = 512):
        self.max_length = max_length

        # Load cleaned data lines
        with open(clean_data_path) as f:

            # Remove empty lines
            self.lines = [lines.strip() for lines in f if lines.strip()]

        # Load tokenizer from file
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Get special token IDs
        self.BOS_id = self.tokenizer.token_to_id("[BOS]")
        self.EOS_id = self.tokenizer.token_to_id("[EOS]")
        self.PAD_id = self.tokenizer.token_to_id("[PAD]")

    def __len__(self):

        # Return number of lines in dataset
        return len(self.lines)
    
    def __getitem__(self , idx):
        
        # Get text line at index
        text = self.lines[idx]

        # Tokenize text to IDs
        id = self.tokenizer.encode(text).ids

        # Add BOS and EOS tokens
        id = [self.BOS_id] + id + [self.EOS_id]

        # Truncate if longer than max length
        if len(id) > self.max_length:
            id = id[:self.max_length]

        # Prepare input IDs, target IDs, and attention mask
        input_ids = id

        # Shift target IDs by one position
        target_ids = id[1:] + [-100]

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len( input_ids )

        # return data sample as dictionary
        return {

            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask
        }
    
# Collate function for DataLoader to pad sequences in batch for making it uniform length , so that torch can process it into tensors
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

