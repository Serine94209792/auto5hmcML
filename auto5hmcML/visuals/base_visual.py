from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_curve_combined(y_val_true, y_val_prob, y_test_true, y_test_prob):
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_prob)

    auc_val = auc(fpr_val, tpr_val)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_val, tpr_val, label=f"Train Set (AUC = {auc_val:.2f})", color="blue")
    plt.plot(fpr_test, tpr_test, label=f"Test Set (AUC = {auc_test:.2f})", color="green")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train & Test Sets)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig("auc.png")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("%s.png" %(title))

def print_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("%s.png" %(title))
