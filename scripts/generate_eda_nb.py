import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction Markdown
    intro_md = """# Exploratory Data Analysis
This notebook explores the cleaned WritingPrompts dataset (25k subset).
We analyze story counts, word count distribution, and basic vocabulary metrics.
"""
    
    # Imports Cell
    imports_cell = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set standard styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""
    
    # Load Data Cell
    load_cell = """import os
# Load Cleaned Data
path1 = '../data/processed/writing_prompts_full_cleaned.parquet'
path2 = 'data/processed/writing_prompts_full_cleaned.parquet'
file_path = path1 if os.path.exists(path1) else path2

df = pd.read_parquet(file_path)
print(f"Loaded {len(df)} stories.")
df.head(2)
"""

    # Word Count Cell
    word_count_cell = """# Calculate Word Counts
df['story_word_count'] = df['clean_story'].apply(lambda x: len(str(x).split()))
df['prompt_word_count'] = df['clean_prompt'].apply(lambda x: len(str(x).split()))

print("Story Word Count Stats:")
print(df['story_word_count'].describe())
"""

    # Plot Distribution Cell
    plot_cell = """# Plot Story Word Count Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['story_word_count'], bins=100, kde=True, color='skyblue')
plt.title('Distribution of Story Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, 2000)  # focusing on the core distribution
plt.tight_layout()
plt.show()
"""

    # Assemble Notebook
    nb.cells = [
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_code_cell(imports_cell),
        nbf.v4.new_code_cell(load_cell),
        nbf.v4.new_code_cell(word_count_cell),
        nbf.v4.new_code_cell(plot_cell)
    ]

    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/01_Exploratory_Data_Analysis.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook created at 'notebooks/01_Exploratory_Data_Analysis.ipynb'")

if __name__ == "__main__":
    create_notebook()
