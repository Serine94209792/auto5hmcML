from typing import List,Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from ..classes import Imbalencer,FirstFilter,SecondFilter,Learner,CustomUnivariateSelect
from sklearn.feature_selection import GenericUnivariateSelect

def RetrieveFeatures(
        pipeline: Pipeline,
        trainset: pd.DataFrame,
        drop_y: Optional[List[str]]=None
        )->None:

    if drop_y is not None:
        trainset=trainset.drop(drop_y,axis=1)

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
