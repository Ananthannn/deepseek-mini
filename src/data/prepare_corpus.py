import os
import re

# save the data that you wanna train the model to the data/raw/raw_data.txt
# cleans the data and saves the cleaned version to data/processed/cleaned_data.txt
class data_processing:

  def __init__(self , raw_data_path , to_save_processed_path):
    self.raw_data_path = raw_data_path
    self.to_save_processed_path = to_save_processed_path

  def read_data(self) -> str:

      # for reading the raw data
      with open(self.raw_data_path , "r" , encoding="utf-8") as f:
          data = f.read()

      return data

  def clean_data(self) -> list:

      text = self.read_data()
      # Replace newlines with spaces
      text = text.replace("\n", " ")

      # Collapse multiple spaces into a single space
      text = " ".join(text.split())

      # Strip leading whitespace
      text = text.strip()

      # Split into sentences or chunks
      lines = re.split(r'\.\s+', text)
      
      cleaned_lines = []
      for line in lines:
          line = line.strip()
          if len(line.split()) >= 5: 
              cleaned_lines.append(line)

      return cleaned_lines

  def save_cleaned_data(self) -> None:
    lines = self.clean_data()
    path = self.to_save_processed_path
    with open(os.path.join(path , "processed.txt"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return None
