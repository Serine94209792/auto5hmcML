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
from typing import Optional
import warnings
import joblib
import os
import numpy as np

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

def plot_roc_curves_all(model_path: str,
                        X: pd.DataFrame,
                        label: str,
                        drop_y: Optional[list] = None,
                        fpr_grid_points: int =100)->None:
    """
    model_path: 传入保存有多个模型的文件路径,
    fpr_grid_points :  FPR 网格点数。
    """
    if label not in X.columns:
        raise ValueError(f"{label} should in X")

    y = X[label]
    y = y.astype(int)
    X = X.drop([label],axis=1)

    if drop_y is not None:
        drop_y = [i for i in drop_y if i != label]
        X = X.drop(drop_y, axis=1)

    models = [i for i in os.listdir(model_path) if os.path.isdir(i)]
    model_dict = {}
    for i in models:
        model_list = [i for i in os.listdir(model_path + "/" + i) if i.endswith(".pkl")]
        model_list = [joblib.load(model_path + "/" + i + "/" + j) for j in model_list]
        model_dict[i] = model_list

    mean_fpr = np.linspace(0, 1, fpr_grid_points)
    plt.figure(figsize=(8, 6))

    for name, model in model_dict.items():
        if not model:
            warnings.warn(f"model '{name}' is empty，check it.")
            continue

        tprs = []
        aucs = []

        for pipeline in model:
            try:
                y_score = pipeline.predict_proba(X)[:, 1]
            except AttributeError:
                y_score = pipeline.decision_function(X)

            fpr, tpr, _ = roc_curve(y, y_score)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc(fpr, tpr))

        tprs = np.vstack(tprs)
        mean_tpr = tprs.mean(axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = tprs.std(axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr,
                 label=f"{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
                 lw=2)
        plt.fill_between(mean_fpr,
                         np.maximum(mean_tpr - std_tpr, 0),
                         np.minimum(mean_tpr + std_tpr, 1),
                         alpha=0.2)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("auc_of_all_models.png")
    plt.close()

def plot_best_roc_curves(model_path: str,
                        X: pd.DataFrame,
                        label: str,
                        drop_y: Optional[list] = None,
                        fpr_grid_points: int =100)->None:
    """
    model_path: 传入保存有多个模型的文件路径,
    fpr_grid_points :  FPR 网格点数。
    """
    if label not in X.columns:
        raise ValueError(f"{label} should in X")

    y = X[label]
    y = y.astype(int)
    X = X.drop([label],axis=1)

    if drop_y is not None:
        drop_y = [i for i in drop_y if i != label]
        X = X.drop(drop_y, axis=1)
    models = [i for i in os.listdir(model_path) if os.path.isdir(i)]
    model_dict = {}
    for i in models:
        model_list = [i for i in os.listdir(model_path + "/" + i) if i.endswith(".pkl")]
        model_list = [joblib.load(model_path + "/" + i + "/" + j) for j in model_list]
        model_dict[i] = model_list

    mean_fpr = np.linspace(0, 1, fpr_grid_points)

    plt.figure(figsize=(8, 6))

    for name, pipelines in model_dict.items():
        best_auc = -np.inf
        best_tpr = None

        if not pipelines:
            warnings.warn(f"model '{name}' is empty，check it.")
            continue

        for pipeline in pipelines:
            try:
                y_score = pipeline.predict_proba(X)[:, 1]
            except AttributeError:
                y_score = pipeline.decision_function(X)

            fpr, tpr, _ = roc_curve(y, y_score)
            this_auc = auc(fpr, tpr)
            if this_auc > best_auc:
                best_auc = this_auc
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                best_tpr = tpr_interp

        plt.plot(mean_fpr, best_tpr, lw=2,
                 label=f"{name} (AUC = {best_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Chance')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_auc_of_all_models.png")
    plt.close()
