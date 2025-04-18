# example of auto5hmcML
![image](image/auto5hmcML.png)

```
from auto5hmcML.models import trainmodels
from auto5hmcML.visuals import Visualization
from auto5hmcML.models import RetrieveFeatures
from auto5hmcML.models import plot_feature_global_logloss_importance,plot_compare_roc
from auto5hmcML.visuals import plot_best_roc_curves, plot_roc_curves_all
import os
import pandas as pd
import numpy as np
import joblib

os.chdir("/home/username/path_to_your_workdir")
dir_path="/home/username/path_to_your_workdir"
trainset = pd.read_csv("trainset.csv")
testset = pd.read_csv("testset.csv")
trainset=pd.read_csv("trainset.csv")
testset=pd.read_csv("testset.csv")
trainlabel = trainset[target_label]
testlabel = testset[target_label]
label = pd.concat([trainlabel, testlabel], axis=0)

trainmodels(trainset=trainset,
            testset=testset,
            y=target_label,
            scoring="accuracy",
            cv=10,
            n_trials=100,
            direction="maximize",
            imbalencer=False,
            n_components_hp=True,
            max_n_compoents=10,
            selection_method="hsic",
            learner_type="pca",
            filter_type="classification",
            ff_kwargs={"mode": "pvalue",
                       "param": 0.05,
                       "function": "kruskal"},
            sf_kwargs={"threshold": 80},
            )

modelpath = "best_moral_type/0_best_model.pkl"
pipeline = joblib.load(modelpath)
    RetrieveFeatures(pipeline,
                     trainset=trainset,
                     testset=testset,
                     y=target_label,
                     )
feature_df = pd.read_csv("second_filter_features.csv")
feature_sel=feature_df["feature_name"].to_numpy().flatten()
Visualization(pipeline,
              trainset,
              testset,
              feature_sel,
              target_label
              )
```

Now the workflow has done!!! you can get the trained models with best hyperparameters, the selected features, and the evaluation plots of the best model. At the same time, the shap_explaination class has been stored in the workdir. We can use it for next model explainaion.

```
shap_values_pca=joblib.load("shap_values_pca.pkl")
plot_compare_roc(shap_values_pca,
                feature='PC4',
                y_true=label)

plot_feature_global_logloss_importance(shap_values_pca, label)

plot_roc_curves_all(model_path=dir_path,
                X=testset,
                label="LM",
                drop_y=training_label,
                fpr_grid_points=100
)

plot_best_roc_curves(model_path=dir_path,
        X=testset,
        label="LM",
        drop_y=training_label,
        fpr_grid_points=100
)
```

