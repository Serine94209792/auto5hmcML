# example of auto5hmcML

```
from auto5hmcML.models import trainmodels
from auto5hmcML.visuals import Visualization
from auto5hmcML.models import RetrieveFeatures
import os
import pandas as pd
import numpy as np
import joblib

os.chdir("/home/username/path_to_your_workdir")
trainset = pd.read_csv("trainset.csv")
testset = pd.read_csv("testset.csv")

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
RetrieveFeatures(pipeline, trainset)
feature_df = pd.read_csv("second_filter_features.csv")
feature_sel=feature_df["feature_name"].to_numpy().flatten()
Visualization(pipeline,
              trainset,
              testset,
              feature_sel,
              target_label
              )
```

Now the workflow has done!!! you can get the trained models with best hyperparameters, the selected features, and the evaluation plots of the best model.