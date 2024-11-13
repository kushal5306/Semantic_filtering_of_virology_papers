import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Load the dataset
df_filtered = pd.read_csv('filtered_results_threshold_0.5.csv')  # Replace with actual path if different

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define descriptions for each method type
method_descriptions = {
    "Text Mining": (
        "Text mining applies natural language processing (NLP) techniques to analyze and extract information from unstructured text data, "
        "such as scientific articles, social media, and clinical reports. This category includes tasks like sentiment analysis, topic discovery, "
        "and entity recognition, aiming to identify patterns related to disease trends, public sentiment, and health information. "
        "It focuses solely on textual data and excludes image or visual data."
    ),
    "Computer Vision": (
        "Computer vision focuses on the analysis of medical and diagnostic imaging, using AI to extract insights from visual data such as radiographs, "
        "CT scans, MRIs, and microscopic images. Applications include image recognition for diagnostics, segmentation, feature extraction, and "
        "screening for diseases. This category is dedicated to image-based data analysis in healthcare, without any text processing components."
    ),
    "Both": (
        "The 'Both' category integrates text mining and computer vision, combining text-based insights with visual data analysis for comprehensive health applications. "
        "This category is relevant for studies involving both textual data (e.g., clinical reports, articles) and image data (e.g., X-rays, pathology images) "
        "to improve diagnostics, predictive modeling, or trend analysis. It includes multi-modal approaches that merge text and image analysis for "
        "disease prediction, diagnostic accuracy, and health insights."
    ),
    "Other": (
        "The 'Other' category includes methods not specifically tied to text mining or computer vision. Examples include statistical modeling, risk prediction, "
        "forecasting, and clinical algorithms that analyze structured data for healthcare purposes, such as disease prediction, epidemiological modeling, "
        "or patient outcome analysis. Methods here may involve numerical data modeling, machine learning for risk assessment, and predictive analysis "
        "focused on clinical decision-making. This category excludes NLP and visual data methods, focusing instead on numeric or structured data analysis."
    )
}


# Generate embeddings for each method description
method_embeddings = {method: model.encode(description) for method, description in method_descriptions.items()}

# Function to classify method type based on similarity
def classify_method_type_with_embeddings(text):
    text_embedding = model.encode(text)
    # Compute similarity between the text embedding and each method description embedding
    similarities = {method: util.cos_sim(text_embedding, embedding).item() for method, embedding in method_embeddings.items()}
    # Find the method with the highest similarity score
    best_method = max(similarities, key=similarities.get)
    return best_method

# Apply the classification to each paper's combined title and abstract
df_filtered['Method_Type'] = df_filtered.apply(
    lambda x: classify_method_type_with_embeddings(f"{x['Title']} {x['Abstract']}" if pd.notna(x['Abstract']) else x['Title']), axis=1
)

# Save the results to a new CSV file
output_file = 'classified_filtered_with_embeddings.csv'
df_filtered.to_csv(output_file, index=False)
print(f"Classified dataset with method types based on sentence embedding similarity saved as '{output_file}'")

# Generate a bar chart to show the distribution of method types
method_counts = df_filtered['Method_Type'].value_counts()

plt.figure(figsize=(10, 6))
method_counts.plot(kind='bar')
plt.title('Distribution of Method Types in Filtered Dataset')
plt.xlabel('Method Type')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.show()