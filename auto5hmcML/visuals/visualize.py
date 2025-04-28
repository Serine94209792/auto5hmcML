import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import umap
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from typing import List
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
from .base_visual import plot_confusion_matrix,plot_roc_curve_combined,print_classification_report
from .classifier_plot import  plot_calibration_curve, plot_ks_curve, plot_cumulative_gain, plot_lift_curve

def Visualization(
        model: Pipeline,
        trainset: pd.DataFrame,
        testset: pd.DataFrame,
        feature_sel: np.ndarray,
        y: str,
        drop_y: List[str],
        )->None:

    if y not in trainset.columns or y not in testset.columns:
        raise ValueError(f"{y} should in trainset and testset")

    trainlabel = trainset[y]
    testlabel = testset[y]
    trainlabel = trainlabel.astype(int)
    testlabel = testlabel.astype(int)
    trainset = trainset.drop([y], axis=1)
    testset = testset.drop([y], axis=1)

    if drop_y is not None:
        drop_y=[i for i in drop_y if i != y]
        trainset=trainset.drop(drop_y,axis=1)
        testset=testset.drop(drop_y,axis=1)

    pred_test=model.predict(testset)
    prob_test=model.predict_proba(testset)[:, 1]
    pred_val=model.predict(trainset)
    prob_val=model.predict_proba(trainset)[:, 1]

    # 生成混淆矩阵
    plot_confusion_matrix(trainlabel, pred_val, "Training Set Confusion Matrix")
    plot_confusion_matrix(testlabel, pred_test, "Testing Set Confusion Matrix")

    # 生成分类报告
    print_classification_report(trainlabel, pred_val, "Training Set Classification Report")
    print_classification_report(testlabel, pred_test, "Testing Set Classification Report")

    # ROC
    plot_roc_curve_combined(trainlabel, prob_val, testlabel, prob_test)

    ##PR curve
    precision_val, recall_val, _ = precision_recall_curve(trainlabel, prob_val)
    precision_test, recall_test, _ = precision_recall_curve(testlabel, prob_test)
    ap_train = average_precision_score(trainlabel, prob_val)
    ap_test = average_precision_score(testlabel, prob_test)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_val, precision_val,
             label=f'Train PR (AP={ap_train:.3f})')
    plt.plot(recall_test, precision_test,
             label=f'Test PR (AP={ap_test:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("PR curve.png")

    dataset=pd.concat([trainset,testset],axis=0)
    label=pd.concat([trainlabel,testlabel],axis=0)
    dataset = dataset.reset_index(drop=True)
    label = label.reset_index(drop=True)
    plot_cumulative_gain(model=model,
                         X=dataset,
                         y=label,
    )

    plot_lift_curve(model=model,
                    X=dataset,
                    y=label,
    )

    plot_calibration_curve(model=model,
                            X=dataset,
                            y=label,
    )

    plot_ks_curve(model=model,
                    X=dataset,
                    y=label,
    )

    label=label.to_list()
    dataset=dataset[feature_sel]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(data_scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=["t-SNE1", "t-SNE2"])
    tsne_df["Group"] = label
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_reducer.fit_transform(data_scaled)
    umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
    umap_df["Group"] = label
    pca_reducer = PCA(n_components=2, random_state=42)
    pca_result = pca_reducer.fit_transform(data_scaled)
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["Group"] = label
    kpca_reducer = KernelPCA(n_components=2, kernel="rbf", random_state=42)
    kpca_result = kpca_reducer.fit_transform(data_scaled)
    kpca_df = pd.DataFrame(kpca_result, columns=["KPC1", "KPC2"])
    kpca_df["Group"] = label

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="t-SNE1", y="t-SNE2", hue="Group", data=tsne_df, palette="tab10", alpha=0.7
    )
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend(title=y, loc="best")
    plt.tight_layout()
    plt.savefig("TSNE.png")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="Group", data=umap_df, palette="tab10", alpha=0.7
    )
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title=y, loc="best")
    plt.tight_layout()
    plt.savefig("UMAP.png")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="PC1", y="PC2", hue="Group", data=pca_df, palette="tab10", alpha=0.7
    )
    plt.title("PCA Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=y, loc="best")
    plt.tight_layout()
    plt.savefig("PCA.png")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="KPC1", y="KPC2", hue="Group", data=kpca_df, palette="tab10", alpha=0.7
    )
    plt.title("KernalPCA Visualization")
    plt.xlabel("KPC1")
    plt.ylabel("KPC2")
    plt.legend(title=y, loc="best")
    plt.tight_layout()
    plt.savefig("KPCA.png")