# Semantic_filtering_of_virology_papers
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C?logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.1%2B-150458?logo=pandas&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-MiniLM--L6--v2-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-blue?logo=matplotlib&logoColor=white)

# Table of Contents

1. [Goals](#goals)
2. [Project Structure & Set Up](#project-structure--set-up)
3. [Step 1: Data Preprocessing](#step-1-data-preprocessing)
4. [Step 2: Semantic Filtering](#step-2-semantic-filtering)
5. [Step 3: Method Type Classification](#step-3-method-type-classification)
6. [Step 4: Method Identification and Classification](#step-4-method-identification-and-classification)


This task aims to filter and classify academic papers from a dataset created through a keyword-based search on PubMed. The dataset is provided in CSV format and contains 11,450 records. The specific goal is to identify papers that implement deep learning neural network-based solutions in virology and epidemiology.
# Goals 
- **Implement NLP-Based Filtering**: Apply semantic natural language processing techniques to filter and exclude papers that do not involve deep learning applications in virology or epidemiology.
  
- **Classify Papers by Method Type**: For relevant papers that meet the criteria, categorize them into one of the following method types:
  - **Text Mining**
  - **Computer Vision**
  - **Both** (using both text mining and computer vision)
  - **Other** (methods not fitting into the above categories)

- **Extract and Report Method Names**: Identify and document the specific deep learning method used in each relevant paper, providing a comprehensive summary of methods employed.

# Project Structure & Set Up
```
├── Code
│   ├── step_1_data_pre_processing
│   │   ├── collection_with_abstracts.csv       # Original dataset file
│   │   ├── data_pre_processing.py              # Script for data preprocessing
│   │   └── preprocessed_dataset.csv            # Output: Preprocessed data
│   ├── step_2_semantic_filtering
│   │   ├── filtered_results_threshold_0.5.csv  # Filtered data at 0.5 threshold
│   │   ├── filtered_results_threshold_0.6.csv  # Filtered data at 0.6 threshold
│   │   ├── filtered_results_threshold_0.7.csv  # Filtered data at 0.7 threshold
│   │   ├── filtered_results_threshold_0.8.csv  # Filtered data at 0.8 threshold
│   │   └── semantic_filtering.py               # Script for semantic filtering
│   ├── step_3_filtering_method_types
│   │   ├── classified_filtered_with_embeddings.csv # Output: Classified by embeddings
│   │   ├── filtered_dataset_stats.png              # Image: Dataset statistics
│   │   └── filtering_method_types.py               # Script for filtering method types
│   └── step_4_extract_methods
│       ├── classified_with_ml_methods.csv          # Final dataset with ML methods
│       ├── extract_methods.py                      # Script to extract ML methods
│       ├── result_method_count_percent.txt         # Text: Method count percentages
│       └── result_method_counts.JPG                # Image: Method counts chart
├── requirements.txt                                # Dependencies for the project
└── README.md                                       # Project documentation

```
Each step of the project will be discussed in detail in the following sections, covering the entire process from data preprocessing and semantic filtering to the classification of method types and extraction of machine learning methods. Additionally, results from each stage will be presented, along with explanations for why specific methods were chosen.

# Step 1: Data Preprocessing

In this step, the raw dataset (`collection_with_abstracts.csv`) is prepared for further analysis. The `data_pre_processing.py` script cleans and processes the data by handling missing values, normalizing text, and removing irrelevant information. The output of this step is a preprocessed dataset (`preprocessed_dataset.csv`) that is ready for semantic filtering and further classification in the subsequent steps.

# Step 2: Semantic Filtering

This step involves filtering the dataset to retain only those papers that utilize deep learning approaches in virology and epidemiology. Instead of relying on keyword-based filtering, which can miss relevant papers due to variations in terminology, this approach uses **semantic similarity** to match each paper against a comprehensive set of prompts representing relevant deep learning applications.

## Methodology
- **Model**: The `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate embeddings for both the prompts and the papers. This model captures semantic meaning, allowing for more accurate matches even when specific keywords are not used. This model is quite small and was quite nice to use in collabs free environment.
- **Prompts**: A wide range of prompts has been created, covering topics like deep learning applications, virology, epidemiology, text mining, computer vision, and multimodal AI approaches. These prompts represent the scope of relevant research areas. This were developed by extensively studying the dataset and by trial and error method to capture as many relevant papers as possible.
- **Threshold Filtering**: Multiple thresholds (0.4, 0.5, 0.6, 0.7, 0.8) are used to filter papers based on their similarity scores. By adjusting these thresholds, we can control the strictness of filtering:
  - **Lower Thresholds** (e.g., 0.4, 0.5): Capture a broader range of potentially relevant papers, but may include more noise.
  - **Higher Thresholds** (e.g., 0.7, 0.8): Provide stricter filtering, retaining only papers with high similarity to the prompts, thus likely more focused on deep learning in virology/epidemiology.
  - Again after studying the resulting dataset I decided that it 0.5 theshold was quite balanced 
## Why This Method is Better Than Keyword Search?
- **Semantic Matching**: Unlike keyword search, which relies on exact terms, semantic similarity allows the model to capture the broader meaning of text. This approach is less dependent on specific words, making it more robust to variations in language. Also it will allow us to capture the deep learning approach in virology and not in the other health areas.
- **Contextual Understanding**: By using a model trained on sentence-level embeddings, this method can understand context, ensuring that only papers relevant to the specified fields are retained, even if they use different terminology.

## Output
For each threshold, a filtered dataset is saved as `filtered_results_threshold_<threshold>.csv`. Each file contains papers that met the respective similarity threshold, including the title, abstract, and similarity score for reference.

The number of entries in each file is as follows:
- **Threshold 0.4**: 5362 entries
- **Threshold 0.5**: 1970 entries
- **Threshold 0.6**: 429 entries
- **Threshold 0.7**: 31 entries
- **Threshold 0.8**: 1 entry
  
- Again after studying the resulting dataset I decided that it 0.5 threshold was quite balanced 

This semantic filtering provides a more flexible and accurate way to identify relevant papers, ensuring that the dataset for further analysis is both comprehensive and focused.

# Step 3: Method Type Classification

This step involves classifying each paper into specific categories based on the type of deep learning method used, again using a semantic similarity approach rather than simple keyword matching. By comparing each paper to predefined method descriptions, our aim is to achieve a more accurate classification.

## Methodology
- **Model**: The `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate embeddings for both the papers and predefined method descriptions. This model captures semantic meaning, which helps accurately categorize each paper.
- **Method Descriptions**: Four categories are defined for classification:
  - **Text Mining**: Involves using NLP for analyzing and extracting information from text data, such as scientific articles or clinical reports.
  - **Computer Vision**: Focuses on analyzing visual data (e.g., radiographs, MRIs) in healthcare contexts.
  - **Both**: Combines text mining and computer vision techniques, analyzing both textual and image data for comprehensive insights.
  - **Other**: Includes statistical modeling, risk prediction, and methods not fitting into the other categories.
- **Classification**: For each paper, the title and abstract are combined to form a single text input, which is then encoded. The similarity between this encoded paper and each method description is calculated, with the highest similarity score determining the method type.

## Advantages of Semantic Similarity Over Keyword Matching
- **Enhanced Precision in Meaning**: Semantic similarity goes beyond literal keywords to capture the underlying meaning of each paper’s content. This allows for accurate classification even if specific keywords are missing or if terms are used in diverse ways across different research contexts.
- **Context-Aware Classification**: By using embeddings, the model assesses each paper within its broader context, enabling a nuanced understanding of the research focus. This reduces the risk of misclassification due to ambiguous or overlapping terminology, ensuring that papers are assigned to the most relevant category based on their actual content. (Atleast this was the idea :))

## Output
The classified dataset is saved as `classified_filtered_with_embeddings.csv`, including the original title and abstract along with the assigned method type. A bar chart is also generated to show the distribution of method types, providing a quick overview of the categorized dataset.

![filtered_dataset_stats](https://github.com/user-attachments/assets/dfdb5f85-22c4-4788-88cb-46ecbe33f6d2)

# Step 4: Method Identification and Classification

In this step, each paper is analyzed to identify the specific machine learning method used. A heuristic-based approach is adopted, utilizing both keyword matching and prioritization rules to assign each paper to a particular ML method. If the method remains unidentified, additional content is fetched from external sources (PubMed Central) to refine the classification further.

## Methodology
- **Keyword-Based Heuristic Matching**: Each paper’s title and abstract are searched for keywords associated with specific ML methods. The list of methods (`ml_methods_priority`) is ordered to prioritize specific, fine-grained methods (e.g., "Convolutional Neural Network" and "Recurrent Neural Network") before broader terms (e.g., "Neural Network" or "Clustering"). This ensures that the most specific ML methods are matched first.
- **Prioritization Rules for Ambiguity Resolution**: To handle cases where multiple methods are detected, a set of `prioritization_rules` is applied. For example:
  - If both "neural network" and "cnn" keywords are found, the method is classified as "Convolutional Neural Network (CNN)."
  - If "transformers" and "attention" are detected, the paper is categorized as using "Transformers."
  - These rules help ensure consistent and accurate classification by resolving overlaps based on context and specificity.

## Additional Processing for Unidentified Methods
- **External Text Fetching for "Unknown" Entries**: For entries that remain classified as "Unknown" after keyword-based matching, the system fetches additional content (up to 500 words) from PubMed Central, using the PMCID. This text is then re-evaluated using the same heuristic approach. If a method is identified in this step, it replaces the initial "Unknown" label; otherwise, the entry is marked as "Still Unknown." This additional step was implemented because large category was classified as unknown after fetching from the PubMed Central, the results were slightly better.

## Advantages of the Heuristic Approach
- **Specificity and Precision**: By prioritizing more specific methods over general ones, the heuristic approach was aimed at improving classification accuracy, ensuring each paper is assigned the most accurate ML method as possible.
- **Efficient Handling of Ambiguity**: Prioritization rules resolve cases where multiple methods appear, enhancing robustness in handling complex or multi-method descriptions. It might not be perfect and there is definetly scope of improvement here.

#### Output
The classified dataset is saved as `classified_with_ml_methods.csv`, with each paper assigned a specific ML method where possible. Additionally, statistics on method counts and percentages are generated, providing insights into the distribution of ML methods within the dataset. Below are the results:

- **Method Counts**:
  - **Still Unknown**: 693
  - **Neural Network**: 458
  - **Unknown**: 298
  - **Random Forest**: 144
  - **Logistic Regression**: 91
  - **Recurrent Neural Network (RNN)**: 70
  - **Gradient Boosting**: 39
  - **Clustering**: 31
  - **Transformers**: 29
  - **Diffusion Models**: 23
  - **Convolutional Neural Network (CNN)**: 22
  - **Support Vector Machine (SVM)**: 19
  - **Autoencoders**: 12
  - **Mathematical Modeling**: 8
  - **Bayesian Networks**: 7
  - **Decision Tree**: 7
  - **Generative Adversarial Network (GAN)**: 5
  - **Principal Component Analysis (PCA)**: 4
  - **K-Means Clustering**: 4
  - **Reinforcement Learning**: 3
  - **Markov Models**: 2
  - **Contrastive Learning**: 1

- **Method Percentages**:
  - **Still Unknown**: 35.18%
  - **Neural Network**: 23.25%
  - **Unknown**: 15.13%
  - **Random Forest**: 7.31%
  - **Logistic Regression**: 4.62%
  - **Recurrent Neural Network (RNN)**: 3.55%
  - **Gradient Boosting**: 1.98%
  - **Clustering**: 1.57%
  - **Transformers**: 1.47%
  - **Diffusion Models**: 1.17%
  - **Convolutional Neural Network (CNN)**: 1.12%
  - **Support Vector Machine (SVM)**: 0.96%
  - **Autoencoders**: 0.61%
  - **Mathematical Modeling**: 0.41%
  - **Bayesian Networks**: 0.35%
  - **Decision Tree**: 0.35%
  - **Generative Adversarial Network (GAN)**: 0.25%
  - **Principal Component Analysis (PCA)**: 0.20%
  - **K-Means Clustering**: 0.20%
  - **Reinforcement Learning**: 0.15%
  - **Markov Models**: 0.10%
  - **Contrastive Learning**: 0.05%

These statistics provide a quick overview of the distribution of ML methods within the dataset, highlighting which methods are most and least common among the classified papers.































