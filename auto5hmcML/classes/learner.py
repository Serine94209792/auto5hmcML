from typing import Optional
from typing import Union
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA, KernelPCA
from gplearn.genetic import SymbolicTransformer



class Learner:
    """
    Learner supports: pca, kpca, umap, SymbolicTransformer
    """
    def __init__(self,
                 n_components: int=2,
                 model_type: str="pca",
                 random_state: int=42,
                 **model_kwargs
                 ):
        """
        params:
        model_type：可选的学习器，kpca,pca,umap,symbolictransformer
        n_components：需要学习的成分个数，可作为超参数
        random_state: int
            随机种子，确保结果可复现。
        model_kwargs: 其它传给基模型的参数
        """
        self.n_components=n_components

        # if self.n_components is None:
        #     raise ValueError(f"n_components must be int !!!!!")

        self.model_type=model_type.lower()
        self.random_state=random_state
        self.model_kwargs=model_kwargs
        self.learner=self._build_learner()


    def _build_learner(self):
        """
        根据 model_type 创建对应的learner
        """
        if self.model_type=="pca":
            learner=PCA(n_components=self.n_components,
                        random_state=self.random_state,
                        **self.model_kwargs)
        elif self.model_type=="umap":
            learner=umap.UMAP(n_components=self.n_components,
                              random_state=self.random_state,
                              **self.model_kwargs)
        elif self.model_type=="kpca":
            learner=KernelPCA(n_components=self.n_components,
                              random_state=self.random_state,
                              **self.model_kwargs)
        elif self.model_type=="symbolictransformer":
            learner=SymbolicTransformer(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.model_kwargs
            )
        else:
            raise ValueError("learner must be one of: "
                             "'pca', 'umap', 'kpca', 'symbolictransformer'")
        return learner

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray]=None):
        """
        kpca,pca,umap无监督，GP有监督
        """
        if self.model_type in ["pca", "umap", "kpca"]:
            self.learner.fit(X)
        else:
            self.learner.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        return self.learner.transform(X)

    def fit_transform(self,
                      X: np.ndarray,
                      y: Optional[np.ndarray] = None) -> np.ndarray:

        return self.learner.fit_transform(X,y)

    def set_params(self, **new_params):
        """
        更新参数
        """
        for key, val in new_params.items():
            setattr(self, key, val)
        self.learner=self._build_learner()
        return self

    def show_feature_relationships(self,
                                   feature_names: Union[str,list[str]]=None
                                   )->pd.DataFrame:
        """
        输出 Learner 模型的输入→输出特征关系:
          - 对 PCA: 打印/返回pca.components_
          - 对 UMAP,kPCA: 返回每个样本的低维坐标
          - 对 SymbolicTransformer: 打印遗传编程生成的表达式

        参数:
        ----------
        feature_names
            如果提供了特征名列表，可在 SymbolicTransformer,pca 的表达式中替换变量名，
        """
        if self.model_type == "pca":
            if not hasattr(self.learner, "components_"):
                raise AttributeError("PCA尚未fit，无法获取components_。请先fit后再调用。")

            comps = self.learner.components_  # shape (n_components, n_features)
            n_components, n_features = comps.shape
            if len(feature_names) == n_features:
                df = pd.DataFrame(comps, columns=feature_names,
                                  index=[f"PC{i + 1}" for i in range(n_components)])
            else:
                df = pd.DataFrame(comps,
                                  index=[f"PC{i + 1}" for i in range(n_components)])
            return df

        elif self.model_type == "umap":
            if not hasattr(self.learner, "embedding_"):
                raise AttributeError("UMAP尚未fit，无法获取embedding_。请先fit后再调用。")
            df = pd.DataFrame(self.learner.embedding_,
                              columns=[f"UMAP{i + 1}" for i in range(self.n_components)])
            return df

        elif self.model_type == "kpca":
            if not hasattr(self.learner, "dual_coef_"):
                raise AttributeError("KPCA 尚未 fit，无法获取 dual_coef_。请先 fit 后再调用。")
            df = pd.DataFrame(self.learner.dual_coef_,
                              columns=[f"KPCA{i + 1}" for i in range(self.n_components)])

            return df

        elif self.model_type == "symbolictransformer":
            if not hasattr(self.learner, "_best_programs"):
                raise AttributeError("SymbolicTransformer尚未fit或内部属性不同，"
                                     "无法获取遗传编程表达式。")

            programs = self.learner._best_programs
            # programs = self.learner._program
            for i, prog in enumerate(programs):
                expr_str = str(prog)
                if feature_names:
                    for idx, fn in enumerate(feature_names):
                        expr_str = expr_str.replace(f"X{idx}", fn)
            return programs

        else:
            raise ValueError("self.model_type 必须是 'pca', 'umap' , ‘kpca', 或 'symbolictransformer'.")
