import os
from tokenizers import Tokenizer, models, trainers

# Train a BPE tokenizer for autoregressive (GPT/DeepSeek-style) training.
class Token_maker:

  def __init__(self , clean_file_path , vocab_path):
    self.clean_file_path = clean_file_path
    self.vocab_path = vocab_path
    self.vocab_size = 20000
    self.BPE_tokenize()

  def BPE_tokenize(self):

      # Initialize tokenizer with BPE model and set unknown token
      tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

      # Special tokens needed for autoregressive LLMs:
      # [PAD] - padding
      # [BOS] - beginning of sequence
      # [EOS] - end of sequence
      # [UNK] - fallback for unknown chars
      trainer = trainers.BpeTrainer(
          vocab_size=self.vocab_size,
          special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
      )

      # Train tokenizer on cleaned dataset
      tokenizer.train([self.clean_file_path], trainer)

      # Save tokenizer JSON
      tokenizer.save(self.vocab_path)