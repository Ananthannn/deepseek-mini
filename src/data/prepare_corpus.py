import os
import re

# save the data that you wanna train the model to the data/raw/raw_data.txt
# cleans the data and saves the cleaned version to data/processed/cleaned_data.txt

def read_data(path: str) -> str:

    # for reading the raw data
    with open(path , "r" , encoding="utf-8") as f:
        data = f.read()

    return data

def clean_data( text:str ) -> list:

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

def save_cleaned_data(lines: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return None

def main():
    #finding the current path
    curr_path = os.path.dirname(os.path.abspath(__file__))

    #finding the root directory
    root_dir = os.path.abspath(os.path.join(curr_path , ".." , ".."))

    #finding the path of the raw text
    raw_data_path = os.path.join(root_dir , "data" , "raw" , "raw_data.txt")

    #finding the path to save the processed data
    processed_data_path =  os.path.join(root_dir , "data" , "processed" , "cleaned_data.txt")
    
    #saving the cleaned data at that path
    save_cleaned_data(clean_data(read_data(raw_data_path)),processed_data_path)


if __name__ == "__main__":
    main()