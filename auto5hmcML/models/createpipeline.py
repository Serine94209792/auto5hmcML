from ..classes import Imbalencer,Learner,FirstFilter,SecondFilter
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional
def CreatePipeline(scaler: bool=True,
                   imbalencer: bool=True,
                   first_filter: bool=True,
                   second_filter: bool=True,
                   learner: bool=True,
                   imbalencer_type: Optional[str]="tomeklinks",
                   selection_method: Optional[str]="l1",
                   filter_type: Optional[str]="svc",
                   learner_type: Optional[str]="pca",
                   n_components: int=2,
                   imbalencer_kwargs: dict=None,
                   ff_kwargs: dict=None,
                   sf_kwargs: dict=None,
                   lr_kwargs: dict=None
                   ) -> Pipeline:
    """
    params:
    scaler 是否需要标准化
    imbalancer 是否需要采样器（针对不平衡数据集）
    first_filter：第一个过滤器为简单过滤，是否需要
    second_filter：第二个过滤器为嵌入式过滤，包括l1正则化法和基于树模型方法，是否需要
    learner: 第三个为特征学习器，是否需要
    imbalencer_type: 如果需要采样器，选用哪种，可选"smote", "kmeans", "centroids", "tomeklinks", "smotetomek"
    selection_method: 如果需要嵌入式过滤器，需要哪种，可选"l1", "tree", "hsic"
    filter_type: 如果需要嵌入式过滤器，需要哪种
        当 selection_method="l1" 时，可选："logistic", "svc"
        当 selection_method="tree" 时，可选："rf", "ada", "xgb"
    learner_type: 如果需要特征学习器，需要哪种
    n_components: 如果需要特征学习器，需要最终学习几个特征,n_components为唯一会被作为超参数的变量!!!!!!
    四个kwargs分别对应四个流程的kwargs

    return:
    返回一个列表，每个元素为一个tuple，每个tuple里第一个为学习器名称，第二个为学习器
    """

    # 如果传入 None 则自动设为空字典，避免后续报错
    if ff_kwargs is None:
        ff_kwargs = {}
    if sf_kwargs is None:
        sf_kwargs = {}
    if lr_kwargs is None:
        lr_kwargs = {}
    if imbalencer_kwargs is None:
        imbalencer_kwargs = {}

    pipeline_list=[]
    if scaler:
        pipeline_list.append(("scaler",StandardScaler()))

    if first_filter:
        pipeline_list.append(("first_filter",FirstFilter(**ff_kwargs)))

    if second_filter:
        pipeline_list.append(("second_filter",SecondFilter(
            selection_method=selection_method,
            model_type=filter_type,
            **sf_kwargs
        )))

    if learner:
        pipeline_list.append(("learner",Learner(
            model_type=learner_type,
            n_components=n_components,
            **lr_kwargs
        )))

    if imbalencer:
        pipeline_list.append(("imbalence", Imbalencer(
            model_type=imbalencer_type,
            **imbalencer_kwargs
        )))

    pipeline=Pipeline(pipeline_list)
    return pipeline
