# 📊 Feature Pair Selection for Cp Prediction

This directory contains the results of a systematic feature-pair selection workflow for predicting heat capacity (Cp) using machine learning.

The goal of this experiment is:

> Select the best combination of two physical features under constraints, and evaluate their predictive power for Cp.

---

## 📁 Directory Structure

2.feature_evaluation_for_initial_data/ 
├── 01_feature_pair_selector.py 
├── config.yaml 
├── mid_project_verified.xlsx 
└── FEATURE_SELECTION_RESULTS/ 
    ├── cleaned_dataset.csv 
    ├── feature_quality_summary.csv 
    ├── single_feature_ranking.csv 
    ├── feature_pair_ranking.csv 
    ├── best_feature_pair_summary.csv 
    ├── best_model_holdout_summary.csv 
    ├── holdout_predictions_best_model.csv 
    ├── best_model_explanation.csv 
    └── *.png (plots) 
 
---

## ⚙️ How to Run

python 01_feature_pair_selector.py --config config.yaml

---

## 🧠 Workflow Overview

### 1. Data Cleaning
- Removes unit rows automatically
- Converts numeric columns
- Drops missing Cp values
- Deduplicates using canonical SMILES

---

### 2. Feature Quality Filtering
Each candidate feature is evaluated based on:
- Non-null ratio
- Variance
- Correlation with Cp

Output:
feature_quality_summary.csv

---

### 3. Single Feature Screening
Each feature is tested independently using multiple models:
- Linear / Ridge / ElasticNet
- SVR
- Random Forest / Gradient Boosting

Metric:
- Repeated CV MAE

Outputs:
single_feature_ranking.csv
single_feature_best_by_feature.csv

---

### 4. Feature Pair Screening
All combinations of 2 features are evaluated.

Includes:
- Cross-validation performance
- Redundancy check
- Information Gain analysis

Outputs:
feature_pair_ranking.csv
top_10_feature_pairs.csv

---

### 5. Information Gain Analysis
Measures how much each feature (or pair) reduces uncertainty in Cp.

Outputs:
information_gain_pair_summary.csv
information_gain_pair_heatmap.png
information_gain_vs_cv_mae.png

---

### 6. Final Model Selection
Best pair + best model is selected based on:
- Lowest CV MAE
- Low redundancy
- High information gain

---

### 7. Holdout Evaluation
Final model is trained and evaluated on a holdout test set.

Outputs:
best_model_holdout_summary.csv
holdout_predictions_best_model.csv
worst_predictions_best_model.csv

---

### 8. Model Interpretation
Includes:
- Permutation importance
- SHAP analysis (if enabled)

Outputs:
best_model_explanation.csv
best_model_shap_beeswarm.png
best_model_shap_bar.png
best_model__dep__*.png

---

## 📈 Key Visualizations

### Model Performance
- top_feature_pairs_mae.png
- feature_count_performance_curve.png

### Data Distribution
- cp_distribution_histogram.png
- cp_distribution_by_compound_type_violin.png

### Correlation
- feature_target_correlation_heatmap.png

### Information Gain
- information_gain_pair_heatmap.png
- information_gain_vs_cv_mae.png

### Final Model
- best_model_holdout_parity.png
- best_model_residual_plot.png

---

## 📊 Final Output Summary

Most important file:

best_feature_pair_summary.csv

Includes:
- Best feature pair
- Best model
- CV performance
- Information gain
- Holdout performance

---

## 🚨 Notes

- SHAP is optional depending on environment
- Data quality strongly affects performance
- Only two features are used due to assignment constraints
