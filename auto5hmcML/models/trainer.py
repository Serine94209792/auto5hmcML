from .createpipeline import CreatePipeline
import pandas as pd
import numpy as np
import optuna
from typing import List, Union, Optional
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from ..utils import get_scores,get_cv_score,optimize_and_build_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import os
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def trainmodels(trainset: pd.DataFrame,
                testset: pd.DataFrame,
                y: str,
                drop_y: Optional[List[str]]=None,
                scoring: Union[str,callable] = "accuracy",
                cv: int = 10,
                n_trials: int = 100,
                direction: str = "maximize",
                scaler: bool=True,
                imbalencer: bool=True,
                first_filter: bool=True,
                second_filter: bool=True,
                learner: bool=True,
                n_components_hp: bool=True,
                n_components: Optional[int]=None,
                max_n_compoents: int=10,
                imbalencer_type: Optional[str]="tomeklinks",
                selection_method: Optional[str]="l1",
                filter_type: Optional[str]="svc",
                learner_type: Optional[str]="pca",
                imbalencer_kwargs: dict = None,
                ff_kwargs: dict = None,
                sf_kwargs: dict = None,
                lr_kwargs: dict = None
                )->None:
    """
    params：训练集测试集，y预测标签，drop_y不需要的标签，预测标签需要在测试机和训练集中
    scoring为交叉验证中的评估参数，默认acc，也可以为自定义scoring
    cv交叉验证折数
    n_trials超参数寻找次数
    direction：优化方向
    当leaner为Ture时，若n_components_hp为True，则n_components为超参数可不提供值，若n_components_hp为False，则n_components必须提供值
    max_n_compoents默认为10，作为超参数的最大搜索值，推荐为训练集样本数的1/10
    kwargs继承自createpipeline

    无返回
    """
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

    if learner and not n_components_hp and n_components is None:
        raise ValueError("when n_components_hp is False, n_components should be provided")

    def objective_svc(trial:optuna.trial)->np.floating:
        C = trial.suggest_loguniform("C", 1e-3, 1)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        ##print(pipe.steps)
        # [("scaler", StandardScaler()), ("smote", SMOTE()), ("clf", LogisticRegression())]
        #print(pipe.named_steps)
        # OrderedDict([
        #   ("scaler", StandardScaler()),
        #   ("smote", SMOTE()),
        #   ("clf", LogisticRegression())
        # ])

        lst=[("svc", SVC(C=C, kernel=kernel, gamma="auto", probability=True,
                        class_weight="balanced",random_state=42))]
        pipeline_svc=pipeline.steps+lst
        pipeline_svc=Pipeline(pipeline_svc)
        val_scores,models=get_cv_score(pipeline_svc, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models",models)
        trial.set_user_attr("val_scores",val_scores)
        test_scores=get_scores(models,testset,testlabel)
        trial.set_user_attr("test_scores",test_scores)
        return np.mean(val_scores)

    def objective_logistic(trial:optuna.trial)->np.floating:
        C = trial.suggest_loguniform("C", 1e-3, 1)
        l1_ratio=trial.suggest_float("l1_ratio",0,1)
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[("logreg", LogisticRegression(
                penalty='elasticnet',
                C=C,
                l1_ratio=l1_ratio,
                solver="saga",
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))]
        pipeline_lr=pipeline.steps+lst
        pipeline_lr=Pipeline(pipeline_lr)
        val_scores,models=get_cv_score(pipeline_lr, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models",models)
        trial.set_user_attr("val_scores",val_scores)
        test_scores=get_scores(models,testset,testlabel)
        trial.set_user_attr("test_scores",test_scores)
        return np.mean(val_scores)

    def objective_rf(trial:optuna.trial)->np.floating:
        n_estimators = trial.suggest_int("n_estimators", 50, 200,step=10)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_split=trial.suggest_int('min_samples_split', 5, 20)  # 内部分裂的最小样本数
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 20) # 叶子节点的最小样本数
        max_samples=trial.suggest_float('max_samples', 0.5, 1.0)  # 每棵树最大样本比例
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[("rf", BalancedRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_samples=max_samples,
                n_jobs=-1,
                sampling_strategy="auto",  #仅采样多数类
                replacement=True,   # 有放回采样
                class_weight="balanced",
                random_state=42
            ))]
        pipeline_rf=pipeline.steps+lst
        pipeline_rf=Pipeline(pipeline_rf)
        val_scores, models = get_cv_score(pipeline_rf, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_mlp(trial:optuna.trial)->np.floating:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            num_units = trial.suggest_int(f"n_units_layer_{i + 1}", 2, 15, step=2)
            hidden_layers.append(num_units)
        alpha = trial.suggest_loguniform("alpha", 1e-4, 1e-1)  # L2正则化系数
        learning_rate_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("mlp", MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=1000,
                batch_size=batch_size,
                random_state=42
            ))
        ]
        pipeline_mlp=pipeline.steps+lst
        pipeline_mlp=Pipeline(pipeline_mlp)
        val_scores, models = get_cv_score(pipeline_mlp, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_xgboost(trial:optuna.trial)->np.floating:
        max_depth = trial.suggest_int("max_depth", 2,6)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-1, 1e1)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-1, 1e1)
        subsample = trial.suggest_uniform("subsample", 0.6, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 50, 200,step=10)
        gamma = trial.suggest_float("gamma",0.0,1.0)
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("xgb", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                eval_metric="logloss",  # 二分类可用logloss
                random_state=42
            ))
        ]
        pipeline_xgb=pipeline.steps+lst
        pipeline_xgb=Pipeline(pipeline_xgb)
        val_scores, models = get_cv_score(pipeline_xgb, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_et(trial:optuna.trial)->np.floating:
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=10)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("et", ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            ))
        ]
        pipeline_et=pipeline.steps+lst
        pipeline_et=Pipeline(pipeline_et)
        val_scores, models = get_cv_score(pipeline_et, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_ada(trial:optuna.trial)->np.floating:
        n_estimators = trial.suggest_int("n_estimators", 50, 200,step=10)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("ada", AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            ))
        ]
        pipeline_ada=pipeline.steps+lst
        pipeline_ada=Pipeline(pipeline_ada)

        val_scores, models = get_cv_score(pipeline_ada, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_gdbt(trial:optuna.trial)->np.floating:
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=10)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
        subsample = trial.suggest_uniform("gbdt_subsample", 0.5, 1.0)
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("gdbt", GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42
            ))
        ]
        pipeline_gdbt = pipeline.steps+lst
        pipeline_gdbt=Pipeline(pipeline_gdbt)
        val_scores, models = get_cv_score(pipeline_gdbt, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    def objective_knn(trial:optuna.trial)->np.floating:
        n_neighbors = trial.suggest_int("n_neighbors", 1, 10)
        weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        nonlocal n_components
        if learner:
            if n_components_hp:
                n_components = trial.suggest_int("n_components", 2, max_n_compoents)
        else:
            n_components = 2

        pipeline = CreatePipeline(
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            n_components=n_components,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs
        )

        lst=[
            ("knn", KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                n_jobs=-1
            ))
        ]
        pipeline_knn=pipeline.steps+lst
        pipeline_knn=Pipeline(pipeline_knn)
        val_scores, models = get_cv_score(pipeline_knn, trainset, trainlabel,scoring,cv)
        trial.set_user_attr("models", models)
        trial.set_user_attr("val_scores", val_scores)
        test_scores = get_scores(models, testset, testlabel)
        trial.set_user_attr("test_scores", test_scores)
        return np.mean(val_scores)

    # def objective_catboost(trial:optuna.trial)->np.floating:
    #     depth = trial.suggest_int("depth", 2, 6)
    #     learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    #     l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-1, 1e2, log=False)
    #     border_count = trial.suggest_int("border_count", 128, 254, step=32)
    #     iterations = trial.suggest_int("iterations",50,300,step=10)
    #     subsample = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
    #     random_strength = trial.suggest_float("random_strength", 0.0, 10.0)
    #     nonlocal pipeline
    #     # has_learner=any(
    #     #     isinstance(step_obj, Learner)
    #     #     for step_name, step_obj in pipeline.steps
    #     # )
    #     if learner:
    #         if pipeline.named_steps["learner"].n_components is None:
    #             n_components_ = trial.suggest_int("n_components", 2, 10)
    #             pipeline.named_steps["learner"].n_components = n_components_
    #
    #     lst=[
    #         CatBoostClassifier(
    #             iterations=iterations,
    #             od_type='Iter',
    #             logging_level='Silent',  # 不打印训练过程
    #             depth=depth,
    #             learning_rate=learning_rate,
    #             l2_leaf_reg=l2_leaf_reg,
    #             border_count=border_count,
    #             subsample=subsample,
    #             random_strength=random_strength,
    #             loss_function="CrossEntropy",
    #             random_state=42
    #         )
    #     ]
    #     pipeline_cb=pipeline.steps+lst
    #     pipeline_cb=Pipeline(pipeline_cb)
    #     # pipeline_cb.steps[-1][1]._estimator_type= "classifier"
    #     val_scores, models = get_cv_score(pipeline_cb, trainset, trainlabel,scoring,cv)
    #     trial.set_user_attr("models", models)
    #     trial.set_user_attr("val_scores", val_scores)
    #     test_scores = get_scores(models, testset, testlabel)
    #     trial.set_user_attr("test_scores", test_scores)
    #     return np.mean(val_scores)

    df_list=[]

    # if not os.path.exists("./catboost/"):
    #     os.mkdir("./catboost/")
    # os.chdir("./catboost/")
    # print("Optimizing catboost...\n")
    # df=optimize_and_build_pipeline(objective_catboost,
    #                             n_trials=n_trials, direction=direction)
    # df.index = ["catboost_" + str(i) for i in df.index]
    # df_list.append(df)
    # os.chdir("../")

    if not os.path.exists("./svc/"):
        os.mkdir("./svc/")
    os.chdir("./svc/")
    print("Optimizing SVC...\n")
    df=optimize_and_build_pipeline(objective_svc,
                                n_trials=n_trials, direction=direction)
    df.index=["svc_"+str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./logistic/"):
        os.mkdir("./logistic/")
    os.chdir("./logistic/")
    print("Optimizing Logistic Regression...\n")
    df=optimize_and_build_pipeline(objective_logistic,
                                n_trials=n_trials, direction=direction)
    df.index=["logistic_"+str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./rf/"):
        os.mkdir("./rf/")
    os.chdir("./rf/")
    print("Optimizing Random Forest...\n")
    df=optimize_and_build_pipeline(objective_rf,
                                n_trials=n_trials, direction=direction)
    df.index = ["rf_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./mlp/"):
        os.mkdir("./mlp/")
    os.chdir("./mlp/")
    print("Optimizing MLP...\n")
    df=optimize_and_build_pipeline(objective_mlp,
                                n_trials=n_trials, direction=direction)
    df.index = ["mlp_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./xgboost/"):
        os.mkdir("./xgboost/")
    os.chdir("./xgboost/")
    print("Optimizing xgBoost...\n")
    df=optimize_and_build_pipeline(objective_xgboost,
                                n_trials=n_trials, direction=direction)
    df.index = ["xgboost_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./et/"):
        os.mkdir("./et/")
    os.chdir("./et/")
    print("Optimizing Extra Trees...\n")
    df=optimize_and_build_pipeline(objective_et,
                                n_trials=n_trials, direction=direction)
    df.index = ["et_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./ada/"):
        os.mkdir("./ada/")
    os.chdir("./ada/")
    print("Optimizing AdaBoost...\n")
    df=optimize_and_build_pipeline(objective_ada,
                                n_trials=n_trials, direction=direction)
    df.index = ["ada_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./gdbt/"):
        os.mkdir("./gdbt/")
    os.chdir("./gdbt/")
    print("Optimizing Gradient Boosting...\n")
    df=optimize_and_build_pipeline(objective_gdbt,
                                n_trials=n_trials, direction=direction)
    df.index = ["gdbt_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    if not os.path.exists("./knn/"):
        os.mkdir("./knn/")
    os.chdir("./knn/")
    print("Optimizing KNN...\n")
    df=optimize_and_build_pipeline(objective_knn,
                                n_trials=n_trials, direction=direction)
    df.index = ["knn_" + str(i) for i in df.index]
    df_list.append(df)
    os.chdir("../")

    df_all=pd.concat(df_list,axis=0)
    df_all.to_csv("all_model_metrics.csv")
