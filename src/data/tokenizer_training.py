import os
from tokenizers import Tokenizer, models, trainers

# Train a BPE tokenizer for autoregressive (GPT/DeepSeek-style) training.
def BPE_tokenize(clean_file_path: str, vocab_path: str, vocab_size: int = 20000):

    # Initialize tokenizer with BPE model and set unknown token
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Special tokens needed for autoregressive LLMs:
    # [PAD] - padding
    # [BOS] - beginning of sequence
    # [EOS] - end of sequence
    # [UNK] - fallback for unknown chars
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    )

    # Train tokenizer on cleaned dataset
    tokenizer.train([clean_file_path], trainer)

    # Save tokenizer JSON
    tokenizer.save(vocab_path)

def main():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(curr_path, "..", ".."))

    clean_file_path = os.path.join(root_dir, "data", "processed", "cleaned_data.txt")
    vocab_path = os.path.join(root_dir, "data", "vocab", "bpe_tokenizer.json")

    BPE_tokenize(clean_file_path, vocab_path)

if __name__ == "__main__":
    main()
