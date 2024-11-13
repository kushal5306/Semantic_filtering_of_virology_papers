import pandas as pd
import re

# Load dataset
df = pd.read_csv('collection_with_abstracts.csv')

# Define a function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

# Apply preprocessing to title and abstract
df['Title'] = df['Title'].apply(preprocess_text)
df['Abstract'] = df['Abstract'].apply(preprocess_text)

# Handle missing abstracts by using title as fallback (optional)
df['Abstract'] = df['Abstract'].fillna(df['Title'])

# Select relevant columns, including PMCID and PMID for future reference
df_preprocessed = df[['Title', 'Abstract', 'PMCID', 'PMID']]

# Save preprocessed dataset for further steps
df_preprocessed.to_csv('preprocessed_dataset.csv', index=False)

print("Preprocessed dataset saved as 'preprocessed_dataset.csv'")
