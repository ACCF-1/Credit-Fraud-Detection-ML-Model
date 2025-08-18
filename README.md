# Imbalance-Classification-ML

A comprehensive machine learning framework for imbalanced classification problems, with a focus on fraud detection. This project provides robust data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and deployment utilities, supporting both standard and panel data.

## Project Purpose

Imbalanced classification is a common challenge in real-world datasets, especially in fraud detection, anomaly detection, and rare event prediction. This repository aims to provide:

- End-to-end ML pipelines for imbalanced datasets.
- Advanced feature engineering and selection techniques.
- Support for a variety of classifiers and resampling methods.
- Automated hyperparameter tuning and model evaluation.
- Tools for model interpretation and visualization.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Imbalance-Classification-ML.git
   cd Imbalance-Classification-ML
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Note:**  
   The project requires the [GeoLite2 Country database](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data) for IP-based feature engineering. Download the `.mmdb` file and place it in the project root or update the path in your code.

## Usage

### Data Preparation

- Place your raw data in the `data/raw/` directory.
- Use the feature engineering scripts in `ml_scripts/feat_engi.py` to preprocess and transform your data.
- Processed data should be saved in `data/processed/`.

### Model Training & Evaluation

- Main modeling logic is in `ml_scripts/modeling.py`.
- Example usage:
  ```python
  from ml_scripts.modeling import main

  # Train and tune an XGBoost model
  main(model_to_train_or_get='xgb', tuning=True, split_ratio=0.2, random_state=42)

  # Load an existing model version (e.g., version 2)
  main(model_to_train_or_get=2)
  ```

- Results, trained models, and evaluation outputs are saved in the `results/` and `saves/` directories.

### Customization

- Model parameters and feature selection strategies can be configured in the `configurations/` directory.
- Extend or modify feature engineering in `ml_scripts/feat_engi.py`.

## Features

- **Imbalanced Data Handling:** SMOTE, ADASYN, and other resampling techniques.
- **Flexible Model Selection:** Supports Random Forest, XGBoost, Logistic Regression, SVM, and more.
- **Feature Engineering:** Includes IP-based, temporal, and categorical feature transformations.
- **Automated Tuning:** Grid search with cross-validation and custom scoring.
- **Interpretability:** SHAP analysis, feature importance plots, and detailed reports.
- **Panel Data Support:** Specialized class for time-series/panel data.

## Requirements

See `requirements.txt` for a full list. Key packages include:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- dill
- geoip2
- matplotlib
- seaborn

## License

This project is licensed under the MIT License.

---
For questions or contributions, please contact Andy via andyinter1@gmail.com.