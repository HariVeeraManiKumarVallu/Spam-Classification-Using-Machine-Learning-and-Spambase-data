import os
import warnings
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, rankdata
import shap
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Dropout, Input  # type: ignore
from keras.optimizers import Adam  # type: ignore
from scikeras.wrappers import KerasClassifier

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=FitFailedWarning)

# -------------------------------
# Configuration
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_FOLDS = 10

# Paths
spambase_data_path = os.path.join(current_dir, 'spambase', 'spambase.data')
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Column names for Spambase (57 features + target)
columns = ["feature_" + str(i) for i in range(1, 58)] + ["is_spam"]

# -------------------------------
# Define Deep Learning Model Function
# -------------------------------
def create_deep_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------
# Define a Feature Processing Pipeline
# -------------------------------
# (Although not wrapped in a scikit-learn Pipeline object, we apply these sequentially)
selector = SelectKBest(f_classif, k=20)
pca = PCA(n_components=0.95)

# -------------------------------
# Define Models Dictionary
# -------------------------------
# For Deep Learning, we set a temporary input_dim of 57 which will be updated after PCA.
MODELS = {
    "Logistic Regression": Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegressionCV(Cs=10, cv=3, class_weight='balanced',
                                      random_state=RANDOM_STATE, max_iter=5000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample',
                                            random_state=RANDOM_STATE),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', gamma='scale', class_weight='balanced',
                    random_state=RANDOM_STATE, probability=True))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                    random_state=RANDOM_STATE),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                                    random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    "Deep Learning": KerasClassifier(model=create_deep_model, model__input_dim=57,
                                     epochs=50, batch_size=32, verbose=0)
}

METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score
}

# -------------------------------
# Helper Function: Extract Feature Importance from Pipeline
# -------------------------------
def extract_feature_importance(model, feature_names):
    """
    If model is a Pipeline and contains a step with feature importance, extract it.
    Otherwise, return None.
    """
    # If model is a pipeline, attempt to extract from the last step.
    if isinstance(model, Pipeline):
        # Try to extract from the last step (assume it's the estimator)
        estimator = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        return np.abs(estimator.coef_[0])
    else:
        return None

# -------------------------------
# Plotting Functions
# -------------------------------
def plot_feature_importance(model, feature_names, save_path):
    importances = extract_feature_importance(model, feature_names)
    if importances is None:
        print(f"Feature importance not available for {model.__class__.__name__}")
        return
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(min(20, len(importances))), importances[indices][:20])
    plt.xticks(range(min(20, len(importances))), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve(estimator, X, y, cv, save_path):
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(.1, 1.0, 5), error_score=np.nan)
        mean_train = np.nanmean(train_scores, axis=1)
        mean_test = np.nanmean(test_scores, axis=1)
        if np.all(np.isnan(mean_train)) or np.all(np.isnan(mean_test)):
            print(f"Learning curve unavailable for {estimator.__class__.__name__}.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, mean_train, 'o-', label="Training score")
        plt.plot(train_sizes, mean_test, 'o-', label="CV score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.title(f"Learning Curve - {estimator.__class__.__name__}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting learning curve for {estimator.__class__.__name__}: {e}")

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cv_distribution(cv_scores, save_path):
    plt.figure(figsize=(10, 6))
    plt.boxplot(list(cv_scores.values()), labels=list(cv_scores.keys()))
    plt.title("CV Score Distribution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def plot_calibration_curve(y_true, y_prob, model_name, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title(f'Calibration Curve - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_shap_summary(model, X, save_path):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"SHAP summary unavailable: {e}")

# -------------------------------
# Function to Process a CV Fold
# -------------------------------
def process_fold(train_index, test_index, X, y):
    fold_results = {}
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Skip fold if training data has only one class
    if len(np.unique(y_train)) < 2:
        print("Skipping fold due to only one class in training data.")
        return fold_results

    for model_name, model in MODELS.items():
        start_time = time.time()
        if model_name == "Deep Learning":
            input_dim = X_train.shape[1]
            model = KerasClassifier(model=create_deep_model, model__input_dim=input_dim,
                                     epochs=50, batch_size=32, verbose=0)
            model.fit(X_train, y_train, validation_split=0.2)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)
        if model_name == "Deep Learning":
            y_pred = (y_pred > 0.5).astype(int)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        fold_results[model_name] = {
            "time": train_time,
            "scores": {metric: fn(y_test, y_pred) for metric, fn in METRICS.items()},
            "roc_auc": roc_auc_score(y_test, proba),
            "y_pred": y_pred.tolist(),
            "y_prob": proba.tolist()
        }
    return fold_results

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Read dataset
    if not os.path.exists(spambase_data_path):
        raise FileNotFoundError(f"Spambase data not found at {spambase_data_path}")
    data = pd.read_csv(spambase_data_path, header=None, names=columns)
    X = data.drop("is_spam", axis=1)
    y = data["is_spam"]

    # Apply feature selection and PCA
    X_selected = selector.fit_transform(X, y)
    X_reduced = pca.fit_transform(X_selected)

    # Update Deep Learning model's input dimension based on reduced features
    input_dim_global = X_reduced.shape[1]
    MODELS["Deep Learning"] = KerasClassifier(model=create_deep_model,
                                                model__input_dim=input_dim_global,
                                                epochs=50, batch_size=32, verbose=0)

    # Run cross-validation folds
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_idx, test_idx, pd.DataFrame(X_reduced), y)
        for train_idx, test_idx in kf.split(X_reduced, y)
    )

    # Aggregate final results
    final_results = {model: {"time": [], **{metric: [] for metric in METRICS},
                             "y_pred": [], "y_prob": []} for model in MODELS}
    for fold in cv_results:
        if not fold:
            continue
        for model, metrics in fold.items():
            final_results[model]["time"].append(metrics["time"])
            for metric, value in metrics["scores"].items():
                final_results[model][metric].append(value)
            final_results[model]["y_pred"].extend(metrics["y_pred"])
            final_results[model]["y_prob"].extend(metrics["y_prob"])

    # Save final results to JSON
    with open(os.path.join(results_dir, 'model_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    # Statistical analysis: Friedman test and model ranking
    try:
        valid_models = [model for model in MODELS if final_results[model]["accuracy"]]
        stat, p_value = friedmanchisquare(*[final_results[model]["accuracy"] for model in valid_models])
    except Exception as e:
        print(f"Friedman test could not be computed: {e}")
        stat, p_value = np.nan, np.nan

    if not np.isnan(p_value) and p_value < 0.05:
        ranks = np.array([rankdata(-np.array(final_results[model]["accuracy"]))
                          for model in valid_models]).mean(axis=1)
        q_critical = 2.343  # For α=0.05, k=8 models
        cd = q_critical * np.sqrt(len(MODELS) * (len(MODELS) + 1) / (6 * len(X)))
        with open(os.path.join(results_dir, 'statistical_analysis.txt'), 'w') as f:
            f.write(f"Significant differences found (critical distance={cd:.3f}):\n")
            for i, model1 in enumerate(MODELS):
                for j, model2 in enumerate(MODELS):
                    if i < j and abs(ranks[i] - ranks[j]) > cd:
                        f.write(f"- {model1} vs {model2}\n")

    # Generate plots for each model
    for model_name, model in MODELS.items():
        # Re-fit model on entire dataset for plotting
        try:
            model.fit(X_reduced, y)
        except Exception as e:
            print(f"Model {model_name} could not be re-fitted: {e}")

        # Plot feature importance if available (skip for Deep Learning)
        if model_name != "Deep Learning":
            plot_feature_importance(model, X.columns, 
                                    os.path.join(results_dir, f'feature_importance_{model_name}.png'))

        plot_learning_curve(model, X_reduced, y, cv=kf,
                            save_path=os.path.join(results_dir, f'learning_curve_{model_name}.png'))
        plot_confusion_matrix(y, final_results[model_name]["y_pred"], model_name,
                              os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))

        if hasattr(model, "predict_proba"):
            optimal_threshold = find_optimal_threshold(y, final_results[model_name]["y_prob"])
            with open(os.path.join(results_dir, f'optimal_threshold_{model_name}.txt'), 'w') as f:
                f.write(f"Optimal threshold: {optimal_threshold:.3f}")
            plot_calibration_curve(y, final_results[model_name]["y_prob"], model_name,
                                   os.path.join(results_dir, f'calibration_curve_{model_name}.png'))

        if model_name == "Random Forest":
            plot_shap_summary(model, X_reduced, os.path.join(results_dir, 'shap_summary.png'))

    plot_cv_distribution({model: final_results[model]["accuracy"] for model in MODELS},
                         os.path.join(results_dir, 'cv_distribution.png'))

    # Performance comparison plot
    metrics_to_plot = ["accuracy", "f1", "time"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    model_names = list(MODELS.keys())
    x_pos = np.arange(len(model_names))
    for idx, metric in enumerate(metrics_to_plot):
        means = [np.mean(final_results[model][metric]) if final_results[model][metric] else np.nan 
                 for model in model_names]
        stds = [np.std(final_results[model][metric]) if final_results[model][metric] else 0 
                for model in model_names]
        axs[idx].bar(x_pos, means, yerr=stds, capsize=5)
        axs[idx].set_title(f"{metric.capitalize()} Comparison")
        axs[idx].set_ylabel(metric.capitalize())
        axs[idx].set_xticks(x_pos)
        axs[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axs[idx].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
    plt.close()

    # Additional: Feature importance comparison across models
    feat_imp = {}
    for model_name, model in MODELS.items():
        imp = extract_feature_importance(model, X.columns)
        if imp is not None:
            feat_imp[model_name] = imp
    if feat_imp:
        plt.figure(figsize=(12, 8))
        for model_name, imp in feat_imp.items():
            plt.plot(range(len(imp)), sorted(imp, reverse=True), label=model_name)
        plt.xlabel('Feature rank')
        plt.ylabel('Feature importance')
        plt.title('Feature Importance Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance_comparison.png'))
        plt.close()

    # ROC Curve Comparison
    plt.figure(figsize=(10, 8))
    for model_name in MODELS:
        if hasattr(MODELS[model_name], "predict_proba"):
            fpr, tpr, _ = roc_curve(y, final_results[model_name]["y_prob"])
            auc_val = np.mean(final_results[model_name]["roc_auc"]) if final_results[model_name]["roc_auc"] else np.nan
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc_curve_comparison.png'))
    plt.close()

    # Model Ranking based on mean accuracy
    model_ranks = {model: np.mean(rankdata(-np.array(final_results[model]["accuracy"]))) if final_results[model]["accuracy"] else np.nan 
                   for model in MODELS}
    ranked_models = sorted(model_ranks.items(), key=lambda x: x[1])
    with open(os.path.join(results_dir, 'model_ranking.txt'), 'w') as f:
        f.write("Model Ranking (based on mean accuracy):\n")
        for rank, (model, score) in enumerate(ranked_models, 1):
            f.write(f"{rank}. {model}: {score:.2f}\n")

    # -------------------------------
    # Print a Summary Table to Console
    # -------------------------------
    print("\n===== Model Performance Summary =====")
    for model in MODELS:
        acc = np.mean(final_results[model]["accuracy"]) if final_results[model]["accuracy"] else np.nan
        f1 = np.mean(final_results[model]["f1"]) if final_results[model]["f1"] else np.nan
        avg_time = np.mean(final_results[model]["time"]) if final_results[model]["time"] else np.nan
        print(f"{model}: Accuracy={acc:.4f}, F1={f1:.4f}, Avg. Training Time={avg_time:.2f}s")
    print("=====================================\n")
    print(f"All results and visualizations have been saved in '{results_dir}'.")
    print("Analysis complete. Check the 'results' directory for detailed outputs.")

