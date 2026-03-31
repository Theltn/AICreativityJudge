import os
from datasets import load_dataset
import pandas as pd

def main():
    print("Starting data collection...")
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    splits = ['validation', 'test']
    
    for split_name in splits:
        print(f"Downloading '{split_name}' split from the 'writingprompts' dataset...")
        try:
            # Load the split
            dataset = load_dataset("euclaise/writingprompts", split=split_name)
            
            # Convert to pandas dataframe
            df = dataset.to_pandas()
            
            # Save to raw data folder
            output_file = f'data/raw/writing_prompts_{split_name}.parquet'
            df.to_parquet(output_file)
            print(f"Successfully downloaded and saved {len(df)} records to {output_file}\n")
            
        except Exception as e:
            print(f"Error downloading {split_name} split: {e}")
if __name__ == "__main__":
    main()
