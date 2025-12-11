import os
from tokenizers import Tokenizer , models , trainers

# Trains a BPE tokenizer on the cleaned data and saves the tokenizer model to the data/vocab as json.
def BPE_tokenize(clean_file_path: str, vocab_path: str, vocab_size: int = 20000):

    # inistializing the tokenizer object with BPE model
    # setting the unknown token to [UNK]

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # setting up the trainer for BPE tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    )

    # training the tokenizer on the cleaned data
    tokenizer.train([clean_file_path], trainer)

    # saving the tokenizer model to the specified path
    tokenizer.save(vocab_path)


def main():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(curr_path, "..", ".."))

    clean_file_path = os.path.join(root_dir, "data", "processed", "cleaned_data.txt")
    vocab_path = os.path.join(root_dir, "data", "vocab", "bpe_tokenizer.json")

    # training the BPE tokenizer and saving the model
    BPE_tokenize(clean_file_path, vocab_path)


if __name__ == "__main__":
    main()