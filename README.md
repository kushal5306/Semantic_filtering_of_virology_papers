## Semantic_filtering_of_virology_papers
The aim of this task is to filter and classify academic papers from a dataset created through a keyword-based search on PubMed. The dataset is provided in CSV format and contains 11,450 records. The specific goal is to identify papers that implement deep learning neural network-based solutions in the fields of virology and epidemiology.

## Project Structure & set up
```plaintext
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

´´´´
iuii



