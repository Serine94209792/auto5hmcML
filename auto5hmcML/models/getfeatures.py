"""
目前该函数在shap时存在问题：
pca后的shap值无法转回pca前，pca固有的损失
不依赖模型的explainer结果不太对
"""

from typing import List,Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from ..classes import Imbalencer,FirstFilter,SecondFilter,Learner,CustomUnivariateSelect
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.decomposition import PCA
import joblib
from scipy.special import logit
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc

def RetrieveFeatures(
        pipeline: Pipeline,
        trainset: pd.DataFrame,
        testset: pd.DataFrame,
        y: str,
        drop_y: Optional[List[str]]=None,
        max_display: int=10,
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

    # has_scaler=any(
    #     isinstance(pipeline_obj, StandardScaler)
    #     for pipeline_name, pipeline_obj in pipeline.steps
    # )
    #
    # has_imbalencer=any(
    #     isinstance(pipeline_obj, Imbalencer)
    #     for pipeline_name, pipeline_obj in pipeline.steps
    # )

    has_first_filter=any(
        isinstance(pipeline_obj,GenericUnivariateSelect) or
        isinstance(pipeline_obj,CustomUnivariateSelect)
        for pipeline_name, pipeline_obj in pipeline.steps
    )

    has_second_filter=any(
        isinstance(pipeline_obj, SecondFilter)
        for pipeline_name, pipeline_obj in pipeline.steps
    )

    has_learner=any(
        isinstance(pipeline_obj, Learner)
        for pipeline_name, pipeline_obj in pipeline.steps
    )

    origin_colnums=np.array(trainset.columns.to_list())

    if has_first_filter:
        # pipeline_sub = pipeline[:3]
        firstfilter=pipeline.named_steps["first_filter"]
        df=pd.DataFrame(
            {
                "feature_name": origin_colnums,
                "score": firstfilter.scores_,
                "pvalue": firstfilter.pvalues_
            }
        )
        selected_f=firstfilter.get_support(indices=True)
        print("the number of first selection is %d" %(len(selected_f)))
        origin_colnums=origin_colnums[selected_f]
        df=df[df["feature_name"].isin(origin_colnums)]
        df.to_csv("first_filter_features.csv")

    if has_second_filter:
        ####所有特征重要性都是指输出特征的重要性，而不是输入！！！！！
        # pipeline_sub = pipeline[:4]
        secondfilter=pipeline.named_steps["second_filter"]
        if secondfilter.selection_method == "hsic":
            secondfilter.selector.save_param("params.csv")
        df = pd.DataFrame(
            {
                "feature_name": origin_colnums,
                "feature_importance": secondfilter.get_feature_importances()
            }
        )
        selected_f = secondfilter.get_support(indices=True)
        print("the number of second selection is %d" % (len(selected_f)))
        origin_colnums = origin_colnums[selected_f]
        df=df[df["feature_name"].isin(origin_colnums)]
        df.to_csv("second_filter_features.csv")

    if has_learner:
        # pipeline_sub = pipeline[:5]
        # X_train,y_train = pipeline_sub.transfrom(trainset,trainlabel)
        # trainset_later = pd.concat([X_train, y_train], axis=0)
        # trainset_later.to_csv("trainset_learner.csv")
        learner=pipeline.named_steps["learner"]
        df=learner.show_feature_relationships(feature_names=origin_colnums)
        df.to_csv("learner_relationship_features.csv")

    steps=pipeline.steps
    dataset = pd.concat([trainset, testset], axis=0)
    label = pd.concat([trainlabel, testlabel], axis=0)
    dataset.reset_index(drop=True, inplace=True)
    label.reset_index(drop=True, inplace=True)

    if has_learner and pipeline.named_steps["learner"].model_type=="pca":
        pca=pipeline.named_steps["learner"]
        pre_pca=Pipeline(steps[:-2])
        model = pipeline.steps[-1][-1]
        datasel=pre_pca.transform(dataset)
        datapca=pca.transform(datasel)
        explainer=shap.Explainer(model.predict_proba,datapca)   ##默认连接函数为概率，可以选择logit
        shap_values=explainer(datapca) #[n_samples, n_features, n_classes]
        shap_values = shap_values[:, :, 1]  ###取正类shap
        ####这里返回一个explainer类，里面有shap值矩阵，base_value向量，输入矩阵三个属性

        pca_feature_names = [f"PC{i + 1}" for i in range(datapca.shape[1])]
        shap_values_pca = shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=pca_feature_names
        )
        joblib.dump(shap_values_pca, "shap_values_pca.pkl")

        shap_mat = shap_values_pca.values
        feature_names = shap_values_pca.feature_names
        mean_abs_shap = np.abs(shap_mat).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })

        importance_df = importance_df.sort_values('mean_abs_shap', ascending=True)
        feature_order = importance_df.index

        shap.plots.beeswarm(shap_values_pca,
                            max_display=max_display)
        plt.tight_layout()
        plt.savefig("beeswarm.png")
        plt.close()

        shap.plots.bar(shap_values_pca,
                       max_display=max_display)
        plt.tight_layout()
        plt.savefig("global_feature_importance.png")
        plt.close()

        select = np.random.choice(datasel.shape[0], size=30, replace=False)
        expected_value = np.unique(shap_values_pca.base_values)
        y_pred = (shap_values_pca.values.sum(1) + expected_value) > 0.5
        ##pca后的值是概率p，因为没填连接函数link
        misclassified = y_pred[select] != label[select]
        shap.decision_plot(base_value=expected_value,
                           shap_values=shap_values_pca[select].values,
                           feature_names=pca_feature_names,
                           feature_order=feature_order.tolist(),
                           feature_display_range=slice(None, None, -1),
                           # link="logit",
                           highlight=misclassified,
                           # new_base_value=np.log(0.5/0.5),
                           # feature_order="hclust",
                           )
        plt.tight_layout()
        plt.savefig("decision_plot_all.png")
        plt.close()

        shap.plots.scatter(shap_values_pca[:, shap_values_pca.abs.mean(0).argsort[-1]],
                               color=shap_values_pca)
        plt.tight_layout()
        plt.savefig(f"scatterplot.png")
        plt.close()

        shap.plots.heatmap(shap_values_pca, max_display=max_display)
        plt.tight_layout()
        plt.savefig("heatmap.png")
        plt.close()

    elif has_learner and pipeline.named_steps["learner"].model_type!="pca":
        pre_decompose=Pipeline(steps[:-2])
        model = Pipeline(steps[-2:])
        datasel = pre_decompose.transform(dataset)
        f = lambda X: model.predict_proba(X)[:, 1]
        background = shap.kmeans(datasel, 100)
        explainer = shap.KernelExplainer(f,background)
        shap_values = explainer(datasel)
        shap_values = shap_values[:, :, 1]

        shap_values = shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=origin_colnums
        )

        joblib.dump(shap_values, "shap_values.pkl")

        shap_mat = shap_values.values
        feature_names = shap_values.feature_names
        mean_abs_shap = np.abs(shap_mat).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=True)
        feature_order = importance_df.index

        shap.summary_plot(shap_values,
                          plot_type="bar",
                          max_display=20,
                          )
        plt.tight_layout()
        plt.savefig("bar.png")
        plt.close()

        shap.plots.beeswarm(shap_values,
                            max_display=20,
                            group_remaining_features=False, )
        plt.tight_layout()
        plt.savefig("beeswarm.png")
        plt.close()

        select = np.random.choice(datasel.shape[0], size=30, replace=False)
        expected_value = np.unique(shap_values.base_values)
        y_pred = (shap_values.values.sum(1) + expected_value) > 0.5
        misclassified = y_pred[select] != label[select]
        shap.decision_plot(base_value=expected_value,
                           shap_values=shap_values[select].values,
                           feature_names=origin_colnums,
                           feature_order=feature_order.tolist(),
                           # link="logit",
                           highlight=misclassified,
                           # new_base_value=np.log(0.5/0.5),
                           # feature_order="hclust",
                           )
        plt.tight_layout()
        plt.savefig("decision_plot.png")
        plt.close()

        shap.decision_plot(base_value=expected_value,
                           shap_values=shap_values[select].values,
                           feature_names=origin_colnums,
                           feature_order=feature_order.tolist(),
                           feature_display_range=slice(None, None, -1),
                           # link="logit",
                           highlight=misclassified,
                           # new_base_value=np.log(0.5/0.5),
                           # feature_order="hclust",
                           )
        plt.tight_layout()
        plt.savefig("decision_plot_all.png")
        plt.close()

        inds = shap.utils.potential_interactions(shap_values[:, shap_values.abs.mean(0).argsort[-1]], shap_values)
        for i in range(3):
            shap.plots.scatter(shap_values[:, shap_values.abs.mean(0).argsort[-1]], color=shap_values[:, inds[i]])
            plt.tight_layout()
            plt.savefig(f"scatter_{i}.png")
            plt.close()

    else:
        pre_process = Pipeline(steps[:-1])
        model = pipeline.steps[-1][-1]
        datasel = pre_process.transform(dataset)
        explainer=shap.Explainer(model.predict_proba,datasel)
        ###没填link默认输出概率
        shap_values=explainer(datasel) #[n_samples, n_features, n_classes]
        shap_values = shap_values[:, :, 1]  ###取正类shap
        joblib.dump(shap_values, "shap_values.pkl")

        shap_mat = shap_values.values
        feature_names = shap_values.feature_names
        mean_abs_shap = np.abs(shap_mat).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=True)
        feature_order = importance_df.index

        shap.summary_plot(shap_values,
                          plot_type="bar",
                          max_display=max_display,
                          )
        plt.tight_layout()
        plt.savefig("bar.png")
        plt.close()

        shap.plots.beeswarm(shap_values,
                            max_display=max_display,
                            group_remaining_features=False)
        plt.tight_layout()
        plt.savefig("beeswarm.png")
        plt.close()

        select = np.random.choice(datasel.shape[0], size=60, replace=False)
        expected_value = np.unique(shap_values.base_values)
        y_pred = (shap_values.values.sum(1) + expected_value) > 0.5
        misclassified = y_pred[select] != label[select]
        shap.decision_plot(base_value=expected_value,
                           shap_values=shap_values[select].values,
                           feature_names=origin_colnums,
                           feature_order=feature_order.tolist(),
                           # link="logit",
                           highlight=misclassified,
                           # new_base_value=np.log(0.5/0.5),
                           # feature_order="hclust",
                           )
        plt.tight_layout()
        plt.savefig("decision_plot.png")
        plt.close()

        shap.decision_plot(base_value=expected_value,
                           shap_values=shap_values[select].values,
                           feature_names=origin_colnums,
                           feature_order=feature_order.tolist(),
                           feature_display_range=slice(None, None, -1),
                           # link="logit",
                           highlight=misclassified,
                           # new_base_value=np.log(0.5/0.5),
                           # feature_order="hclust",
                           )
        plt.tight_layout()
        plt.savefig("decision_plot_all.png")
        plt.close()

        inds = shap.utils.potential_interactions(shap_values[:, shap_values.abs.mean(0).argsort[-1]], shap_values)
        for i in range(3):
            shap.plots.scatter(shap_values[:, shap_values.abs.mean(0).argsort[-1]], color=shap_values[:, inds[i]])
            plt.tight_layout()
            plt.savefig(f"scatter_{i}.png")
            plt.close()

def simulate_missing_feature(shap_expl:shap.Explanation,
                             )->pd.DataFrame:
    """
    模拟去除每个特征时的预测概率

    params:
    ----
    shap_expl : shap.Explanation

    returns:
    ----
    probs_df : pd.Dataframe,
        对应每个样本，缺失每一个特征时的预测概率。
    """
    shap_mat = shap_expl.values
    base = shap_expl.base_values

    # 如果 base 是标量，扩展成与样本数一致的向量
    if np.isscalar(base):
        base = np.full(shap_mat.shape[0], base)

    sum_all = shap_mat.sum(axis=1)  # shape = (n_samples,)

    n_samples, n_features = shap_mat.shape
    probs = np.zeros((n_samples, n_features))
    for j in range(n_features):
        sum_without_j = sum_all - shap_mat[:, j]
        logit_missing = base + sum_without_j
        probs[:, j] = expit(logit_missing)

    index = getattr(shap_expl, 'data_index', None)
    probs_df = pd.DataFrame(probs,
                            index=index,
                            columns=shap_expl.feature_names)
    return probs_df

def feature_global_logloss_importance(shap_expl:shap.Explanation,
                               y_true:pd.Series)->pd.Series:
    """
    计算完整模型与去除每个特征后对数损失（交叉熵损失）差值

    params:
    ----
    shap_expl : shap.Explanation
    y_true : label

    returns:
    ----
    pd.Series，为每个特征对所有样本的交叉熵损失
        索引为各特征名，值为log_loss_without_feature − log_loss_full
        >0为移除该特征后模型性能下降，<0为移除该特征后模型性能提升。
    """

    shap_mat = shap_expl.values
    base = shap_expl.base_values
    if np.isscalar(base):
        base = np.full(shap_mat.shape[0], base)

    logits_full = base + shap_mat.sum(axis=1)
    probs_full  = expit(logits_full)
    loss_full   = log_loss(y_true, probs_full)

    losses = {}
    for j, feat in enumerate(shap_expl.feature_names):
        logits_no = base + (shap_mat.sum(axis=1) - shap_mat[:, j])
        probs_no  = expit(logits_no)
        loss_no   = log_loss(y_true, probs_no)
        #移除后 − 原始
        losses[feat] = loss_no - loss_full

    return pd.Series(losses).sort_values(ascending=False)

def feature_local_logloss_importance(shap_expl:shap.Explanation,
                               y_true:pd.Series)->pd.DataFrame:
    """
    计算每个样本、去除每个特征后的 log‑loss 相对于完整模型的差值矩阵。

    返回
    ----
    pd.DataFrame, shape=(n_samples, n_features)
    """
    shap_mat = shap_expl.values
    base = shap_expl.base_values
    if np.isscalar(base):
        base = np.full(shap_mat.shape[0], base)

    logits_full = base + shap_mat.sum(axis=1)        # (n_samples,)
    probs_full  = expit(logits_full)                 # (n_samples,)
    eps = 1e-15                                      # 防止 log(0)
    lf = -(y_true * np.log(probs_full + eps) +
           (1 - y_true) * np.log(1 - probs_full + eps))  # (n_samples,)

    n_samples, n_features = shap_mat.shape
    diffs = np.zeros((n_samples, n_features))
    for j in range(n_features):
        logits_no_j = base + (shap_mat.sum(axis=1) - shap_mat[:, j])
        probs_no_j  = expit(logits_no_j)
        ln = -(y_true * np.log(probs_no_j + eps) +
               (1 - y_true) * np.log(1 - probs_no_j + eps))
        diffs[:, j] = ln - lf

    index = getattr(shap_expl, 'data_index', None)
    if index is None:
        index = np.arange(n_samples)
    df = pd.DataFrame(diffs,
                      index=index,
                      columns=shap_expl.feature_names)
    return df

def plot_compare_roc(shap_expl:shap.Explanation,
                     feature:str,
                     y_true:pd.Series)->None:
    """
    在同一张图中比较完整模型与去除指定特征后的 ROC 曲线，并保存图片。

    参数
    ----
    shap_expl : shap.Explanation
    simulate_missing_feature : callable之前定义的函数
    feature : str
        要“删除”的特征名
    y_true : label
    """
    base = shap_expl.base_values
    if np.isscalar(base):
        base = np.full(shap_expl.values.shape[0], base)
    logits_full = base + shap_expl.values.sum(axis=1)
    probs_full  = expit(logits_full)
    fpr_full, tpr_full, _ = roc_curve(y_true, probs_full)
    auc_full = auc(fpr_full, tpr_full)

    # 2. 计算去除指定特征后的概率
    probs_no = simulate_missing_feature(shap_expl)
    fpr_no, tpr_no, _ = roc_curve(y_true, probs_no[feature])
    auc_no = auc(fpr_no, tpr_no)

    # 3. 绘制并保存
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_full, tpr_full, label=f'With {feature}    (AUC = {auc_full:.4f})')
    plt.plot(fpr_no,    tpr_no,    label=f'Without {feature} (AUC = {auc_no:.4f})')
    plt.plot([0, 1],    [0, 1],     '--', color='grey', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison (remove "{feature}")')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'auc_comparison_remove {feature}.png', dpi=300)
    plt.close()

def plot_feature_global_logloss_importance(shap_expl: shap.Explanation,
                                           y_true: pd.Series,
                                           top_n: int = 20)->None:
    """
    绘制柱状图

    params:
    ----
    shap_expl : shap.Explanation
        SHAP 解释对象
    y_true : pd.Series
        真实标签（0/1）
    top_n : int
        只显示绝对重要性排名前 top_n 的特征，默认显示前20个
    """
    # 1. 计算全局重要性并排序
    importances = feature_global_logloss_importance(shap_expl, y_true)
    # 按绝对值降序
    ordered = importances.abs().sort_values(ascending=False)
    ordered = ordered.iloc[:top_n]
    ordered_imp   = importances.loc[ordered.index]

    # 2. 绘图
    n_feats = len(ordered_imp)
    plt.figure(figsize=(8, max(4, n_feats * 0.3)))
    plt.barh(ordered_imp.index, ordered_imp.values)
    plt.xlabel('Δ Log Loss (no feature − full model)')
    plt.ylabel('Features')
    plt.title(f'Global Log Loss Importance per Feature (top {n_feats})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("log_loss_difference", dpi=300)
    plt.close()


