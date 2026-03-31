import os
import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Remove markdown artifacts like headers (##), bold (**), blockquotes (>), etc.
    # Replace bold/italic
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    
    # Remove Reddit tags like [ WP ], [ EU ], [ RF ], etc.
    text = re.sub(r'\[\s*[a-zA-Z]{2}\s*\]\s*', '', text)
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_story(text, min_words=100):
    # Split by whitespace for simple word count
    words = text.split()
    return len(words) >= min_words

def preprocess_dataset(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    initial_count = len(df)
    print(f"Initial story count: {initial_count}")
    
    # Clean texts
    print("Cleaning text (removing URLs, markdown, reddit tags)...")
    df['clean_story'] = df['story'].apply(clean_text)
    df['clean_prompt'] = df['prompt'].apply(clean_text)
    
    # Filter by minimal length to remove broken/too short stories
    print("Filtering short stories (< 100 words)...")
    valid_mask = df['clean_story'].apply(is_valid_story)
    df_clean = df[valid_mask].copy()
    
    final_count = len(df_clean)
    print(f"Final valid story count: {final_count} (removed {initial_count - final_count})")
    
    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_parquet(output_path)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess WritingPrompts data')
    parser.add_argument('--input', type=str, default='data/raw/writing_prompts_full.parquet', help='Input parquet file')
    parser.add_argument('--output', type=str, default='data/processed/writing_prompts_full_cleaned.parquet', help='Output parquet file')
    
    args = parser.args() if hasattr(parser, "args") else parser.parse_args()
    preprocess_dataset(args.input, args.output)
