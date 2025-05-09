import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn_pandas import DataFrameMapper
from typing import Tuple, Union, List
from sklearn.ensemble import VotingClassifier
import optuna
import joblib
from joblib import parallel_backend
import warnings
import os
import gc
import json

def SplitDataSet(expre_input,dataset_input,dataset_output,target,test_size=0.2):
    """
    params:
    expre_input为输入表达矩阵
    dataset_input为输入临床信息
    dataset_output为清洗后的输出特征矩阵路径（表达+临床）
    target为要分层的特征
    test_size为测试集大小

    return:
    trainset, testset
    """
    expre=pd.read_csv(expre_input,index_col=0)
    expre=np.log1p(expre)
    expre = expre.transpose()
    expre["sample"] = expre.index.to_list()
    dataset=pd.read_csv(dataset_input)
    drops = ["LN+", "Nstage", "LNR","type",
             "RetrievedLN", "max_tumor_size","margin"]
    dataset=dataset.drop(drops,axis=1)
    clinic_num=dataset.shape[1]
    dataset=pd.merge(dataset,expre,how="inner",on="sample")
    numer = ["age", "log2_ca199"]
    cate = ["sex", "LM","smoke", "drink", "dm",
             "pancreatitis", "differentiation",
            "VascularInvasion", "NeuralInvasion", "AdiposeInvasion"]
    genes=dataset.columns[clinic_num:]
    mapper1 = [([cols], [SimpleImputer(missing_values=np.nan, strategy="median")]) for cols in numer]
    mapper2 = [([cols], [SimpleImputer(missing_values=np.nan, strategy="most_frequent")]) for cols in cate]
    mapper3 = [([cols], None) for cols in genes]
    mapper = DataFrameMapper(mapper1 + mapper2+ mapper3, input_df=True, df_out=True)
    ###先划分再归一
    trainset, testset = train_test_split(dataset, test_size=test_size, stratify=dataset[target], shuffle=True,random_state=42)
    trainsetsample = trainset["sample"]
    trainsetsample.to_csv("trainsetsample.csv",header=True,index=False)
    trainset = mapper.fit_transform(trainset)
    testset = mapper.fit_transform(testset)
    trainset.to_csv("trainset.csv",header=True,index=False)
    testset.to_csv("testset.csv", header=True, index=False)
    dataset.to_csv(dataset_output,header=True,index=False)
    mapper1 = [([cols], [SimpleImputer(missing_values=np.nan, strategy="median"), StandardScaler()]) for cols in numer]
    mapper2 = [([cols], [SimpleImputer(missing_values=np.nan, strategy="most_frequent")]) for cols in cate]
    mapper3 = [([cols], [StandardScaler()]) for cols in genes]
    mapper = DataFrameMapper(mapper1 + mapper2 + mapper3, input_df=True, df_out=True)
    trainset_scale = mapper.fit_transform(trainset)
    trainset_scale.to_csv("trainset_scale.csv", header=True, index=False)
    return trainset,testset

def get_cv_score(pipeline: Union[Pipeline,VotingClassifier],
                 X: pd.DataFrame,
                 y: pd.Series,
                 scoring: Union[str,callable]="accuracy",
                 cv_splits: int=10,
                 n_jobs: int=-1,
                 )->Tuple[np.ndarray,np.ndarray]:

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,random_state=42)
    cv_results = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs,return_estimator=True)
    ####使用包装好的cross_validata方便加速
    val_scores = cv_results["test_score"]
    models = cv_results["estimator"]
    return val_scores,models

def get_scores(models:np.ndarray,
               X:pd.DataFrame,
               y:pd.Series)->List[Tuple]:
    scores=[]
    i=0
    for model in models:
        i+=1
        scores.append(evaluations(model,X,y))
    return scores

def evaluations(model,
                X: pd.DataFrame,
                y:pd.Series)->Tuple[float,float,float,float]:
    y_pred=model.predict(X)
    accuracy=accuracy_score(y,y_pred)
    f1=f1_score(y,y_pred)
    mcc=matthews_corrcoef(y,y_pred)
    Y_pred_prob=model.predict_proba(X)[:,1]
    auc=roc_auc_score(y,Y_pred_prob)
    return accuracy,f1,mcc,auc

def optimize_and_build_pipeline(
        objective_func,
        n_trials: int =100,
        direction: str="maximize")->pd.DataFrame:

    gc.collect()
    study = optuna.create_study(direction=direction)
    study.optimize(objective_func, n_trials=n_trials)
    print("Best value:", study.best_trial.values)
    print("Best params:", study.best_trial.params)
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.write_image("opt_history.png")
    best_models=study.best_trial.user_attrs['models']
    best_val_scores=study.best_trial.user_attrs['val_scores']
    best_test_scores=study.best_trial.user_attrs['test_scores']
    colnames=["val_score"]
    colnames+=["test_"+i for i in ["acc","f1","mcc","auc"]]
    colnames=np.asarray(colnames)
    colnames=colnames.flatten()
    shape_row=len(best_models)
    df=pd.DataFrame(index=range(shape_row),columns=colnames)
    i = 0
    for model in best_models:
        joblib.dump(model, str(i) + "_best_model.pkl")
        df.loc[i,colnames[0]]=best_val_scores[i]
        df.loc[i,colnames[1:]]=best_test_scores[i]
        i+=1

    mean_row=df.mean()
    df.loc["mean"]=mean_row
    df.to_csv("metrics.csv")
    return df

def load_top_n_estimators(
        metric_file: str = "all_model_metrics.csv",
        top_n: int = 5,
        json_out: str = "selected_models.json"
) -> List[Tuple[str, object]]:
    """
    读取 metrics，按 Val_score 降序选出前 n_top 个，
    返回 [(model_name, estimator), …]。
    行索引示例： logistic_0, xgboost_3 …；含mean的行自动忽略。
    """
    df = pd.read_csv(metric_file,index_col=0)
    df = df.loc[~df.index.str.contains(r"mean", case=False)]
    top_idx = df["val_score"].sort_values(ascending=False).head(top_n).index
    estimators = []
    path_mapping={}
    for name in top_idx:
        family, idx = name.rsplit("_", 1)          # 拆成文件夹名和编号
        pkl_path = os.path.join(family, f"{idx}_best_model.pkl")
        if not os.path.exists(pkl_path):
            warnings.warn(f"cannot find {pkl_path}, skip this model")
            continue
        est = joblib.load(pkl_path)
        estimators.append((name, est))
        path_mapping[name]=pkl_path

    if len(estimators)==0:
        raise RuntimeError("no estimators found, please check the model files")

    if json_out is not None:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(path_mapping, f, indent=4, ensure_ascii=False)

    return estimators