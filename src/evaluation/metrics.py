import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)

def plot_roc_curve(y_true, y_pred_prob, model_name):
    """Plot ROC curve for the given model."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob, model_name):
    """Plot Precision-Recall curve for the given model."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"{model_name} (AP = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.show()

def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred, digits=4))

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def evaluate_model(y_true, y_pred, y_pred_prob, model_name):
    """Run full evaluation pipeline for a given model."""
    print(f"📊 Model: {model_name}")
    
    # Metrics
    print_classification_report(y_true, y_pred)
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_prob):.4f}")
    
    # Plots
    plot_roc_curve(y_true, y_pred_prob, model_name)
    plot_precision_recall_curve(y_true, y_pred_prob, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
