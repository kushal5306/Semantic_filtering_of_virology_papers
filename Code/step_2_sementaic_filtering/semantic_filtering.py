import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Define threshold values to test
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
results = {}

# Load preprocessed dataset
df = pd.read_csv('preprocessed_dataset.csv')

# Load model on the selected device, was trained on free gpu in google collab
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Define the full list of prompts
prompts = [
    # general Prompts
    "Deep learning applications in virology.",
    "Using CNN or RNN for virus detection.",
    "NLP techniques in epidemiology research.",
    "Applications of deep learning in virology.",
    "Predictive modeling of viral outbreaks using deep neural networks.",
    "Use of LSTM networks for viral spread prediction.",
    "Epidemiological data analysis using machine learning models.",
    "Deep neural networks for detecting viral patterns in data.",
    "Application of convolutional neural networks in virus detection.",
    "Using RNNs for epidemiological forecasting.",
    "Transformer models for virus sequence analysis.",
    "Generative models for simulating viral transmission.",
    "Analyzing epidemiological trends with autoencoders.",
    "Natural language processing techniques for epidemiology research.",
    "Extracting virus-related information using text mining.",
    "Analyzing scientific literature on virology using NLP.",
    "Text mining of virology papers for information extraction.",
    "Language models for understanding virology literature.",
    "Using machine learning for diagnosing viral infections.",
    "Forecasting virus spread with machine learning techniques.",
    "AI applications for analyzing virology research data.",
    "Use of neural networks for studying epidemiology of viruses.",
    "Understanding viral evolution with deep learning models.",
    "Sequence-to-sequence models in virology.",
    "Attention mechanisms for analyzing viral genome sequences.",
    "Multimodal models for combining virology data sources.",
    "Using diffusion models in epidemiological simulations.",
    "Deep learning for virus image classification and recognition.",
    "Integrating computer vision and text mining for virus detection.",
    "Multimodal AI models for epidemiological data analysis.",
    "Combining NLP and computer vision in virology research.",
    "Fusion of text and image data for viral outbreak monitoring.",

    # Expanded Prompts for Virology and Epidemiology
    "Machine learning applications for virus detection and classification.",
    "Predictive modeling for viral infection spread.",
    "Deep learning approaches in virology.",
    "Using neural networks to classify viral pathogens.",
    "Applications of convolutional neural networks in virus imaging.",
    "Machine learning models for tracking viral transmission.",
    "Analyzing viral genome sequences with deep learning.",
    "Using artificial intelligence to monitor viral outbreaks.",
    "Modeling virus-host interactions using machine learning.",
    "Predictive models for analyzing viral load and progression.",
    "Machine learning in epidemiological forecasting for infectious diseases.",
    "Predictive modeling for pandemic and epidemic preparedness.",
    "Artificial intelligence applications in epidemiology.",
    "Deep learning for tracking disease spread in populations.",
    "Using machine learning for disease surveillance and outbreak detection.",
    "Data-driven approaches to studying infectious disease transmission.",
    "Using neural networks to model epidemiological patterns.",
    "AI in public health surveillance of infectious diseases.",
    "Machine learning models for epidemic outbreak prediction.",
    "Deep learning for identifying transmission chains in populations.",
    "Using machine learning to study coronavirus spread and mutations.",
    "AI applications in COVID-19 epidemiology and virus classification.",
    "Deep learning for analyzing SARS-CoV-2 transmission data.",
    "Predicting influenza outbreaks with neural networks.",
    "Machine learning in HIV and other viral infection research.",
    "Predictive modeling of emerging viral infections.",
    "Deep learning models for herpesvirus and other viral genomes.",
    "Machine learning applications in respiratory virus tracking.",
    "Data analysis for viral infections using artificial intelligence.",
    "Epidemiological analysis of infectious diseases with AI.",
    "Image processing for diagnosing viral infections.",
    "Computer vision applications in CT and MRI for virus detection.",
    "Machine learning for interpreting diagnostic imaging of viruses.",
    "Analyzing diffusion-weighted imaging data for viral infection insights.",
    "Deep learning in diagnostic imaging for viral disease progression.",
    "Neural networks for identifying viral patterns in diagnostic scans.",
    "Predictive modeling of viral infections from medical imaging data.",
    "Using machine learning to process CT scans in epidemiological studies.",
    "Data-driven imaging models for virus diagnosis and prognosis.",
    "AI in CDC Emerging Infections Program for infectious disease tracking.",
    "Machine learning in public health surveillance of viral outbreaks.",
    "Data analytics for tracking viral infections in populations.",
    "Predictive modeling in infectious disease surveillance systems.",
    "Using machine learning for CDC viral infection data analysis.",
    "AI approaches for emerging infections program in epidemiology.",
    "Machine learning for monitoring viral prevalence in public health."
]

# Generate embeddings for prompts
prompt_embeddings = model.encode(prompts, convert_to_tensor=True).to(device)

for threshold in thresholds:
    filtered_papers = []

    for _, row in df.iterrows():
        paper_text = f"{row['Title']}. {row['Abstract']}" if pd.notna(row['Abstract']) else row['Title']
        paper_embedding = model.encode(paper_text, convert_to_tensor=True).to(device)
        max_similarity = util.cos_sim(paper_embedding, prompt_embeddings).max().item()

        # Apply threshold to filter papers
        if max_similarity >= threshold:
            filtered_papers.append({
                'Title': row['Title'],
                'Abstract': row['Abstract'],
                'PMCID': row['PMCID'],
                'PMID': row['PMID'],
                'Similarity_Score': max_similarity
            })

    # Store results for this threshold
    results[threshold] = pd.DataFrame(filtered_papers)

# Save each threshold's results separately for review
for threshold, df_result in results.items():
    df_result.to_csv(f'filtered_results_threshold_{threshold}.csv', index=False)
    print(f"Results saved for threshold {threshold} in 'filtered_results_threshold_{threshold}.csv'")
