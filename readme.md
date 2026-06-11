![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Association%20Rules-orange)
![IoT](https://img.shields.io/badge/IoT-Research-green)
![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)
![Version](https://img.shields.io/badge/Version-v1.1.0-blueviolet)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)

# IoT Event Analysis and Decision Support System

This project implements a complete framework for exploratory analysis, synthetic data expansion, and association rule mining of IoT-generated events.

The framework was developed to support research on human-in-the-loop response behavior in IoT monitoring systems, enabling the discovery of temporal and operational patterns associated with alert generation and operator intervention.

---

## Objective

To develop a reproducible framework for exploratory analysis and association rule mining of IoT-generated behavioral events, enabling the discovery of temporal and operational patterns associated with human response dynamics.

The framework supports:

* Temporal event analysis
* Synthetic dataset expansion
* Transactional dataset generation
* Association rule mining
* Experimental comparison of ARM algorithms
* Decision-support research in IoT-based monitoring systems

---

## System Overview

The project is structured into five main components:

* **Deterministic Pipeline** → Data cleaning and feature engineering
* **Synthetic Expansion Pipeline** → Controlled generation of synthetic records
* **EDA Module** → Statistical and visual exploratory analysis
* **ARM Module** → Association Rule Mining
* **Experiment Tracking Module** → Reproducible evaluation and comparison of algorithms

---

## Methodological Workflow

The complete workflow follows the sequence:

1. Data loading
2. Data cleaning and preprocessing
3. Synthetic dataset expansion
4. Temporal feature engineering
5. Feature discretization
6. Transactional dataset generation
7. Exploratory Data Analysis (EDA)
8. Association Rule Mining (Apriori, FP-Growth, ECLAT)
9. Experimental evaluation and tracking
10. Export of rules and metrics

---

## Pipeline Design

The deterministic pipeline follows a structured and reproducible workflow:

1. Data loading
2. Datetime parsing
3. Data cleaning
4. Synthetic dataset expansion
5. Temporal feature engineering
6. Event dynamics modeling
7. Rolling window feature generation
8. Feature discretization
9. Transactional conversion
10. Association rule mining

---

## Key Features Generated

### Response Features

* Response time (`response_seconds`)
* Response time class (`response_time_class`)

### Temporal Features

* Hour (`hour`)
* Minute (`minute`)
* Minute of day (`minute_of_day`)
* Day of week (`day_of_week`)
* Weekend indicator (`is_weekend`)
* Time period (`part_of_day`)

### Event Dynamics Features

* Time between events (`time_since_prev_alert_s`)
* Alerts within the last 15 minutes (`alerts_last_15m`)
* Alerts within the last 60 minutes (`alerts_last_60m`)

---

## Exploratory Data Analysis (EDA)

The project includes statistical and visualization tools for:

### Statistical Analysis

* Mean
* Median
* Mode
* Standard deviation
* Quartiles
* Interquartile range (IQR)
* Minimum and maximum values

### Visual Analysis

* Response time distribution (Histogram + KDE)
* Response time dispersion (Boxplot)
* Response rate by hour
* Weekday versus weekend comparison
* Correlation matrix

---

## Synthetic Dataset Expansion

Considering the limited availability of real-world operational events, the framework includes a controlled synthetic expansion process.

The synthetic generation pipeline:

* Preserves attendance proportions
* Preserves temporal distributions
* Preserves response time distributions
* Preserves operational characteristics
* Maintains logical consistency between variables

Generated outputs include:

* Expanded dataset
* Synthetic-only dataset
* Statistical validation report

The synthetic dataset is used exclusively for methodological evaluation and algorithm comparison, not as a replacement for real-world validation.

---

## Association Rule Mining (ARM)

The framework implements three classical Association Rule Mining algorithms:

### Apriori

Generates frequent itemsets through candidate generation and iterative support pruning.

### FP-Growth

Uses an FP-Tree structure to efficiently mine frequent patterns without candidate generation.

### ECLAT

Uses a vertical transaction representation and set intersections to identify frequent itemsets.

The algorithms are executed using the same transactional dataset and comparable hyperparameters, allowing a fair experimental comparison.

---

## Rule Evaluation Metrics

The generated rules are evaluated using:

### Support

Frequency of occurrence of a rule within the dataset.

### Confidence

Conditional probability of the consequent given the antecedent.

### Lift

Strength of association relative to statistical independence.

---

## Experimental Tracking

The framework automatically records:

* Dataset type
* Number of transactions
* Number of unique items
* Minimum support
* Minimum confidence
* Minimum lift
* Minimum itemset size
* Number of frequent itemsets
* Number of generated rules
* Number of filtered rules
* Execution time
* Operating system
* Python version

---

## Generated Outputs

### Processed Datasets

* Cleaned datasets
* Expanded datasets
* Feature-engineered datasets
* Discretized datasets
* Transactional datasets

### EDA Outputs

* Histograms
* Boxplots
* Attendance-rate plots
* Correlation matrices
* Descriptive statistics reports

### ARM Outputs

For each algorithm:

* Association rules (.csv)
* Support
* Confidence
* Lift

### Experimental Evaluation Outputs

* ARM experiment summary
* Execution time comparison
* Rule distribution statistics
* Algorithm comparison metrics

---

## Project Structure

```
.
├── config/
│   └── config.py
│
├── datasets/
│   ├── 0_observational_dataset.csv
│   ├── 1_observational_cleaned_preprocessed_dataset.csv
│   ├── 2_synthetic_expanded_dataset.csv
│   ├── 2_synthetic_expansion_validation_report.csv
│   ├── 2_synthetic_records_only.csv
│   ├── 3_observational_feature_engineering_dataset.csv
│   ├── 3_synthetic_feature_engineering_dataset.csv
│   ├── 4_observational_discretization_categorical_dataset.csv
│   ├── 4_synthetic_discretization_categorical_dataset.csv
│   ├── 5_observational_transactional_dataset.csv
│   └── 5_synthetic_transactional_dataset.csv
│
├── eda/
│   ├── statistics_eda.py
│   └── visualization_eda.py
│
├── outputARM/
│   ├── apriori_observational_output.csv
│   ├── apriori_synthetic_output.csv
│   ├── fpgrowth_observational_output.csv
│   ├── fpgrowth_synthetic_output.csv
│   ├── eclat_observational_output.csv
│   ├── eclat_synthetic_output.csv
│   ├── arm_experiment_summary_observational.csv
│   └── arm_experiment_summary_synthetic.csv
│
├── pipelines/
│   ├── arm_apriori_pipeline.py
│   ├── arm_fpgrowth_pipeline.py
│   ├── arm_eclat_pipeline.py
│   ├── arm_experiment_tracking_pipeline.py
│   ├── cleaning_preprocess_pipeline.py
│   ├── discretization_categorical_pipeline.py
│   ├── evaluation_metrics_pipeline.py
│   ├── feature_engineering_pipeline.py
│   ├── synthetic_dataset_expansion_pipeline.py
│   └── transactional_format_pipeline.py
│
├── .gitignore
├── license.md
├── main.py
├── readme.md
└── requirements.txt
```

---

## Methodological Considerations

* Deterministic pipeline ensures reproducibility
* Temporal ordering is preserved
* Synthetic expansion is statistically controlled
* Transactional conversion supports ARM algorithms
* Multiple ARM algorithms are available for comparison
* Experimental tracking ensures reproducibility
* Rule evaluation focuses on interpretability and operational relevance

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

The source code may be used, modified, and shared for academic and non-commercial purposes, provided that appropriate credit is given to the author.

The dataset used in this study is not publicly available due to confidentiality restrictions.

---

## Author

Developed by **Bruna Sousa**.

Electrical Engineer | IoT Developer | Web Developer | AI Developer

GitHub: https://github.com/bruna-sousa-dev

LinkedIn: https://www.linkedin.com/in/bruna-sousa-dev

Email: [brunampsousa.dev@gmail.com](mailto:brunampsousa.dev@gmail.com)
