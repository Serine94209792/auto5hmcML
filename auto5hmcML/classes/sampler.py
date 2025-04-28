from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, TomekLinks
from imblearn.combine import SMOTETomek
import numpy as np
from typing import Tuple

class Imbalencer:
    """
    Imbalencer supports:
      - over_sampling strategies: SMOTE, kmeansSMOTE
      - under_sampling strategies: ClusterCentroids,TomekLinks
      _ combination: SMOTETomek
    """
    def __init__(self,
                 model_type: str="tomeklinks",
                 random_state: int =42,
                 **model_kwargs):
        """
        参数：
        model_type : str
            "smote", "borderline", "centroids", "tomeklinks", "smotetomek"

        random_state: int
            随机种子，确保结果可复现。

        model_kwargs: 其它传给基模型的参数
            如 SMOTE 的 sampling_strategy, k_neighbors 等。
        """
        self.model_type=model_type.lower()
        self.random_state=random_state
        self.model_kwargs=model_kwargs
        self.sampler=self._build_sampler()

    def _build_sampler(self):
        """
        根据 model_type 创建对应的采样器
        """
        if self.model_type == "smote":
            sampler = SMOTE(random_state=self.random_state, **self.model_kwargs)

        elif self.model_type == "borderline":
            sampler = BorderlineSMOTE(random_state=self.random_state, **self.model_kwargs)

        elif self.model_type == "centroids":
            sampler = ClusterCentroids(random_state=self.random_state, **self.model_kwargs)

        elif self.model_type == "tomeklinks":
            sampler = TomekLinks(**self.model_kwargs)
            # TomekLinks不需要random_state

        elif self.model_type == "smotetomek":
            sampler = SMOTETomek(random_state=self.random_state, **self.model_kwargs)

        else:
            raise ValueError("sampler must be one of: "
                             "'smote', 'kmeans', 'centroids', 'tomek', 'smotetomek'")

        return sampler

    def fit(self,X: np.ndarray, y: np.ndarray):
        """
        对输入数据 X, y 执行重采样。
        """
        self.sampler.fit(X, y)
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对输入数据 X, y 执行重采样。
        """
        X_res, y_res = self.sampler.fit_resample(X, y)
        return X_res, y_res

    def set_params(self, **new_params):
        """
        动态更新采样器的参数；若需要在采样器创建后修改其配置，可调用此方法。
        """
        for key, val in new_params.items():
            setattr(self, key, val)
        self.sampler=self._build_sampler()
        return self
