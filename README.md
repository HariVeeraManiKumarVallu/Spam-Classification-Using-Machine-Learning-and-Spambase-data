**Spam Classification Using Machine Learning and Spambase Data:**

**About the Repository**
This repository presents a comprehensive spam classification project using a range of machine learning models—including deep learning—applied to the Spambase dataset from the UCI Machine Learning Repository.  The project leverages advanced feature selection, dimensionality reduction, cross-validation, and statistical analysis to evaluate and compare the performance of eight different classifiers.

**Project Overview**

Spam emails are a persistent challenge, and accurately classifying them is essential for effective email filtering.  In this project:

- **Dataset:**  The Spambase dataset contains 4,601 email instances (with 39.4% spam) and 58 attributes (57 continuous features plus 1 nominal class label).  

 - **Models Evaluated:**
   - Logistic Regression (with polynomial features)
   - Random Forest
   - Support Vector Machine (SVM)  
   - Gradient Boosting
   - Neural Network (MLP)  
   - XGBoost
   - LightGBM
   - Deep Learning (Keras-based neural network)

- **Feature Engineering:**
  - **SelectKBest:** ANOVA F-test to choose the top 20 features.  
   - **PCA:** Applied to retain 95% of the variance.
  
 - **Evaluation:**
   - **10-Fold Stratified Cross-Validation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
    
 - **Visualizations & Analysis:**  
   - Feature Importance plots, Learning Curves, Confusion Matrices, Calibration Curves, ROC Curve comparisons, CV Score Distributions, and Performance Comparison charts.
   - Statistical analysis using the Friedman test to rank models.
   - SHAP analysis for interpretability (applied to the Random Forest model).


**About the Code:**

The main script, `spam_classification.py`, performs the following tasks:
 1. **Data Loading & Preprocessing:** - Reads the Spambase dataset from `spambase/spambase.data` (with accompanying documentation and names files).  
    - Applies SelectKBest and PCA for feature selection and dimensionality reduction.
 2. **Model Training & Evaluation:**  
    - Defines eight models in a dictionary, including a deep learning model wrapped via Keras.  
    - Runs 10-fold cross-validation using `StratifiedKFold` and parallel processing (via `joblib`), computing metrics (accuracy, F1, etc.) for each fold.
 3. **Visualization & Statistical Analysis:**  
    - Generates multiple plots (e.g., feature importance, learning curves, confusion matrices, ROC curves, calibration curves, CV distribution, and performance comparisons).  
    - Performs a Friedman test to statistically compare model performances and produces model rankings.
 4. **Output:**  
    - All outputs (JSON reports, plots, ranking files) are saved in the `results/` directory.


## Outputs Generated
 After running the script, the following outputs are saved in the `results/` folder:
 - **Performance Metrics:**  
   - `model_results.json`: Aggregated evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, and training time).  
   - `model_ranking.txt`: Models ranked by mean accuracy.  
   - `statistical_analysis.txt`: Friedman test results and model comparisons.
  
 - **Visualizations:**
   - `feature_importance_<model>.png`: Feature importance plots (for models that provide them). 
   - `learning_curve_<model>.png`: Learning curves for each model.  
   - `confusion_matrix_<model>.png`: Confusion matrices.  
   - `calibration_curve_<model>.png`: Calibration curves for models with probability outputs.  
   - `roc_curve_comparison.png`: ROC curve comparison across models.  
   - `cv_distribution.png`: Boxplot showing the distribution of cross-validation scores.  
   - `performance_comparison.png`: Bar chart comparing accuracy, F1-score, and training time.

 - **Threshold Optimization & SHAP Analysis:**
   
   - `optimal_threshold_<model>.txt`: The computed optimal threshold for probability-based models.  
   - `shap_summary.png`: SHAP summary plot (applied to the Random Forest model).


## Spambase Dataset Documentation 
- **Title:** SPAM E-mail Database
- **Creators:** Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt *(Hewlett-Packard Labs, Palo Alto, CA)
  - **Donor:** George Forman
  - **Generated:** June–July 1999
  - ** Usage:**  
   - Originally used in Hewlett-Packard’s internal research for spam filtering.
   - Helps determine whether an email is spam with a misclassification error of ~7%.
 - **Attributes:**  
   - 57 continuous features indicating word and character frequencies and run-length measures.
   - 1 nominal class label indicating spam (1) or non-spam (0).
 - **Instances:**  
   - 4,601 emails (1,813 spam [39.4%] and 2,788 non-spam [60.6%]).
 - **Dataset Link:** [UCI Spambase Dataset] (https://archive.ics.uci.edu/ml/datasets/spambase)


**How to Run the Project:**

**1) Clone the repository:**

git clone 

cd spam-classification

**2) Install required dependencies:**

pip install -r requirements.txt

**3) Run the classification script:**

python spam_classification.py

4) Check the results/ directory for outputs.

**Contact Information**

For any questions or collaborations, feel free to reach out:

📧 Email: manikumarvallu@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/hari-veera-mani-kumar-vallu-073625210/

📂 GitHub: https://github.com/HariVeeraManiKumarVallu
