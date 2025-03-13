**Spam Classification Using Machine Learning:**

**About the Repository:**
This repository contains a comprehensive spam classification project using multiple machine learning models, including deep learning. The project utilizes the Spambase dataset to classify emails as spam or non-spam. Various feature selection, preprocessing techniques, and cross-validation strategies are employed to ensure robust model performance evaluation.

**Project Overview:**
1) Dataset: Spambase dataset
2) Models Used: Logistic Regression, Random Forest, SVM, Gradient Boosting, Neural Network, XGBoost, LightGBM, Deep Learning (Keras)
3) Feature Selection: SelectKBest (ANOVA F-test), PCA (95% variance retained)
4) Evaluation Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC
5) Plots & Visualizations: Feature Importance, Learning Curves, Confusion Matrices, ROC Curves, Calibration Curves, CV Score Distributions
6) Statistical Analysis: Friedman test for model comparison

**About the Code:**

The main script, spam_classification.py, performs the following tasks:

1) Loads the Spambase dataset and preprocesses it.
2) Applies feature selection and dimensionality reduction using SelectKBest and PCA.
3) Trains multiple machine learning models using 10-fold cross-validation.
4) Computes evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
5) Generates visualizations like feature importance plots, learning curves, confusion matrices, and ROC curves.
6) Conducts statistical analysis to compare models and find the best performer.
7) Saves results in a results/ directory, including JSON reports, performance comparison charts, and ranking summaries.

**Outputs Generated:**

After running spam_classification.py, the following files and visualizations are saved in the results/ directory:

**Performance Metrics:**

1) model_results.json - Contains accuracy, precision, recall, F1-score, ROC-AUC, and training time for all models.
2) model_ranking.txt - Lists models ranked by mean accuracy.
3) statistical_analysis.txt - Friedman test results for model comparisons.

**Visualizations:**
1) feature_importance_<model>.png - Feature importance plots for tree-based models.
2) learning_curve_<model>.png - Learning curves showing model generalization.
3) confusion_matrix_<model>.png - Heatmap visualization of classification performance.
4) roc_curve_comparison.png - Comparison of ROC curves across models.
5) calibration_curve_<model>.png - Calibration curves for probability-based models.
6) cv_distribution.png - Boxplot of cross-validation score distribution.
7) performance_comparison.png - Bar charts comparing accuracy, F1-score, and training time.

**Threshold Optimization & SHAP Analysis:**
1) optimal_threshold_<model>.txt - The best decision threshold for classification.
2) shap_summary.png - SHAP analysis of Random Forest model for feature interpretability.

**Spambase Dataset:**

The Spambase dataset used in this project can be found at the following link:
🔗 UCI Machine Learning Repository - Spambase Dataset: http://www.ics.uci.edu/~mlearn/MLRepository.html

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
