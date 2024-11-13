# Semantic_filtering_of_virology_papers
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

### Step 2: Semantic Filtering

This step involves filtering the dataset to retain only those papers that utilize deep learning approaches in virology and epidemiology. Instead of relying on keyword-based filtering, which can miss relevant papers due to variations in terminology, this approach uses **semantic similarity** to match each paper against a comprehensive set of prompts representing relevant deep learning applications.

#### Methodology
- **Model**: The `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate embeddings for both the prompts and the papers. This model captures semantic meaning, allowing for more accurate matches even when specific keywords are not used. This model is quite small and was quite nice to use in collabs free environment.
- **Prompts**: A wide range of prompts has been created, covering topics like deep learning applications, virology, epidemiology, text mining, computer vision, and multimodal AI approaches. These prompts represent the scope of relevant research areas. This were developed by extensively studying the dataset and by trial and error method to capture more relevant papers 
- **Threshold Filtering**: Multiple thresholds (0.4, 0.5, 0.6, 0.7, 0.8) are used to filter papers based on their similarity scores. By adjusting these thresholds, we can control the strictness of filtering:
  - **Lower Thresholds** (e.g., 0.4, 0.5): Capture a broader range of potentially relevant papers, but may include more noise.
  - **Higher Thresholds** (e.g., 0.7, 0.8): Provide stricter filtering, retaining only papers with high similarity to the prompts, thus likely more focused on deep learning in virology/epidemiology.
  - Again after studying the resulting dataset I decided that it 0.5 theshold was quite balanced 
#### Why This Method is Better Than Keyword Search
- **Semantic Matching**: Unlike keyword search, which relies on exact terms, semantic similarity allows the model to capture the broader meaning of text. This approach is less dependent on specific words, making it more robust to variations in language. Also it will allow us to capture the deep learning approach in virology and not in the other health areas.
- **Contextual Understanding**: By using a model trained on sentence-level embeddings, this method can understand context, ensuring that only papers relevant to the specified fields are retained, even if they use different terminology.

#### Output
For each threshold, a filtered dataset is saved as `filtered_results_threshold_<threshold>.csv`. Each file contains papers that met the respective similarity threshold, including the title, abstract, and similarity score for reference.

The number of entries in each file is as follows:
- **Threshold 0.4**: 5362 entries
- **Threshold 0.5**: 1970 entries
- **Threshold 0.6**: 429 entries
- **Threshold 0.7**: 31 entries
- **Threshold 0.8**: 1 entry
  
- Again after studying the resulting dataset I decided that it 0.5 threshold was quite balanced 

This semantic filtering provides a more flexible and accurate way to identify relevant papers, ensuring that the dataset for further analysis is both comprehensive and focused.
































