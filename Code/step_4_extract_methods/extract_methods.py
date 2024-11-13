import pandas as pd
import re
import time
import requests
import matplotlib.pyplot as plt

# Step 3: Load the dataset
df = pd.read_csv('classified_filtered_with_embeddings (1).csv')

# Step 4: Define the refined list of ML methods without broad terms
ml_methods_priority = {
    "Convolutional Neural Network (CNN)": ["convolutional neural network", "cnn"],
    "Recurrent Neural Network (RNN)": ["recurrent neural network", "rnn", "gru", "gated recurrent unit", "lstm", "long short-term memory"],
    "Support Vector Machine (SVM)": ["support vector machine", "svm"],
    "K-Means Clustering": ["k-means clustering", "kmeans", "clustering", "unsupervised learning"],
    "Random Forest": ["random forest", "ensemble learning"],
    "Logistic Regression": ["logistic regression"],
    "Neural Network": ["neural network", "ann", "artificial neural network", "mlp", "multilayer perceptron"],
    "Bayesian Networks": ["bayesian networks", "bayesian inference", "bayesian optimization"],
    "Decision Tree": ["decision tree"],
    "Gradient Boosting": ["gradient boosting", "xgboost", "lightgbm", "adaboost", "boosting"],
    "Transformers": ["transformers", "bert", "gpt", "attention mechanism", "attention-based neural network", "sequence-to-sequence"],
    "Generative Adversarial Network (GAN)": ["gan", "generative adversarial network"],
    "Autoencoders": ["autoencoder", "variational autoencoder", "vae"],
    "Markov Models": ["markov chain", "hidden markov model", "markov decision process"],
    "Principal Component Analysis (PCA)": ["pca", "principal component analysis"],
    "Clustering": ["clustering", "unsupervised clustering", "hierarchical clustering", "dbscan", "fuzzy clustering"],
    "Diffusion Models": ["diffusion model", "diffusion-based generative model"],
    "Contrastive Learning": ["contrastive learning"],
    "Few-Shot Learning": ["few-shot learning", "meta-learning"],
    "Reinforcement Learning": ["reinforcement learning", "q-learning", "policy gradient", "deep reinforcement learning"],
    "Mathematical Modeling": ["mathematical modeling", "numerical analysis", "stochastic analysis", "epidemiological modeling"]
}

# Step 5: Define prioritization rules for common overlaps and hybrid approaches
prioritization_rules = {
    ("neural network", "cnn"): "Convolutional Neural Network (CNN)",
    ("neural network", "rnn"): "Recurrent Neural Network (RNN)",
    ("clustering", "k-means"): "K-Means Clustering",
    ("boosting", "gradient boosting"): "Gradient Boosting",
    ("ensemble learning", "random forest"): "Random Forest",
    ("transformers", "attention"): "Transformers",
    ("recurrent neural network", "lstm"): "Recurrent Neural Network (RNN)",
    ("unsupervised", "clustering"): "Clustering",
    ("reinforcement learning", "policy gradient"): "Reinforcement Learning",
    ("diffusion model", "generative"): "Diffusion Models",
    ("few-shot learning", "meta-learning"): "Few-Shot Learning",
    ("gan", "adversarial"): "Generative Adversarial Network (GAN)",
    ("mathematical modeling", "numerical analysis"): "Mathematical Modeling",
    ("deep learning", "reinforcement learning"): "Reinforcement Learning with Deep Learning"
}

# Step 6: Define the function to identify ML methods based on keywords and prioritization rules
def identify_ml_method(title, abstract):
    text = f"{title.lower()} {abstract.lower()}"
    detected_methods = set()

    # Search for each method in the refined priority list
    for method, keywords in ml_methods_priority.items():
        if any(re.search(rf"\b{keyword}\b", text) for keyword in keywords):
            detected_methods.add(method)

    # Apply prioritization rules if multiple methods are found
    if len(detected_methods) > 1:
        for rule, prioritized_method in prioritization_rules.items():
            if all(r in detected_methods for r in rule):
                return prioritized_method

    # Return the most specific or first detected method, or "Unknown" if none found
    return detected_methods.pop() if detected_methods else "Unknown"

# Step 7: Apply the function to each row to create a new column for ML methods
df['ML_Method'] = df.apply(lambda x: identify_ml_method(x['Title'], x['Abstract']), axis=1)

# Step 8: Fetch additional content for "Unknown" entries if possible, and attempt reclassification
def fetch_article_text(pmcid):
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
        response = requests.get(url)
        if response.status_code == 200:
            article_text = response.text
            first_500_words = ' '.join(article_text.split()[:500])
            return first_500_words
    except Exception as e:
        print(f"Error fetching text for PMCID {pmcid}: {e}")
    return None

# Reclassify "Unknown" entries with additional text from PMC
for idx, row in df[df['ML_Method'] == 'Unknown'].iterrows():
    pmcid = row.get('PMCID')
    if pd.notna(pmcid):
        additional_text = fetch_article_text(pmcid)
        if additional_text:
            new_method = identify_ml_method(row['Title'], additional_text)
            df.at[idx, 'ML_Method'] = new_method if new_method != "Unknown" else "Still Unknown"
    time.sleep(1)  # Rate limiting for the API

# Step 9: Generate and display data statistics for analysis
method_counts = df['ML_Method'].value_counts()

# Display statistics
print("ML Method Counts:")
print(method_counts)

method_percentage = (method_counts / len(df)) * 100
print("\nML Method Percentages:")
print(method_percentage)

# Step 10: Save results to a CSV file
df.to_csv('classified_with_ml_methods.csv', index=False)
print("Saved the results to 'classified_with_ml_methods.csv'")
