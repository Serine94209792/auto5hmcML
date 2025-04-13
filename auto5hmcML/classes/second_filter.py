from pyHSICLasso import HSICLasso
from typing import Union,Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


class HSICLassoTransformer(HSICLasso, BaseEstimator, TransformerMixin):
    """
    多重继承，sklearn/Transformer+HSICLasso。
    这样就能在使用Pipeline时调用 fit / transform，
    同时也能直接调用 HSICLasso 里已有的方法（dump, get_index, plot_path 等）。
    """

    def __init__(self,
                 mode="classification",  # or "regression"
                 num_feat=20,
                 B=0,
                 M=1,
                 discrete_x=False,
                 max_neighbors=5,
                 n_jobs=-1,
                 covars=np.array([]),
                 covars_kernel="Gaussian"):
        """
        这里把HSICLasso相关的主要参数都列出来，
        让HSICLassoTransformer在实例化时可设置。
        """
        # 先调用父类 HSICLasso 的初始化，确保其内部属性都准备好了
        HSICLasso.__init__(self)

        # 下面这些是包装器本身需要的属性
        self.mode = mode
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.feature_importances_ = None    # 存放输入特征的重要性
        self.selected_idx_ = None  # 用于在 transform() 阶段知道选了哪些列
        self.n_features_ = None  # 记录fit时的特征总数

    def fit(self, X, y=None):
        """
        符合 sklearn fit() 规范：
        调用classification或regression
        记录选出的特征索引
        记录[输入特征]的重要性
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if y is None:
            raise ValueError("HSICLasso需要监督信息，y不可为空。")
        y = np.asarray(y).ravel()

        X_in,Y_in=X,y
        self.input(X_in, Y_in)   # 调用HSICLasso的 input 方法

        # 根据 mode 调用 HSICLasso 的分类或回归
        if self.mode == "classification":
            self.classification(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        elif self.mode == "regression":
            self.regression(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        else:
            raise ValueError("mode必须是 'classification' 或 'regression'。")

        # -------------- 3) 记录选中列索引 ---------------
        # HSICLasso 会把最终选出的特征存到 self.A
        # self.A => (selected_features, ) (索引)
        # 由于行=特征，所以 self.A 正好对应 sklearn 里的特征列
        self.selected_idx_ = self.A

        # -------------- 4) 计算并存储 feature_importances_ ---------------
        # 做法：创建一个长度 n_features 的 0 数组，对选中的特征给出非0分数
        # 例如 (beta[i][0] / maxval) 来自 HSICLasso dump() 的思路
        feature_importances = np.zeros(n_features)

        # 如果 self.beta 存在、且 A 非空，就可做标准化
        # 例如 dump() 里: maxval = self.beta[self.A[0]][0]
        # 注意 self.beta 是一个2D数组: shape(#features, ?)，看它实际情况(跟选择的邻近neighbor数目有关)
        if len(self.A) > 0 and self.beta is not None:
            maxval = float(self.beta[self.A[0]][0])  # 取第一个被选中特征的 beta，beta降序排列，第一个特征beta最大
            for idx in self.A:
                feature_importances[idx] = float(self.beta[idx][0]) / maxval

        self.feature_importances_ = feature_importances

        return self

    def transform(self, X):
        """
        只保留HSICLasso选中的特征列
        """
        if self.selected_idx_ is None:
            raise RuntimeError("必须先调用fit()再transform()。")

        X = np.asarray(X)
        return X[:, self.selected_idx_]

    def get_support(self, indices: bool = False):
        """
        indices=True，返回一个索引数组，比如 [0, 2, 5...].
        indices=False，返回一个长度为 n_features_ 的布尔数组，选中的特征对应 True。
        """
        if self.selected_idx_ is None:
            raise ValueError("HSICLassoTransformer还未fit，无法调用get_support")

        if indices:
            return self.selected_idx_
        else:
            # 生成布尔mask
            mask = np.zeros(self.n_features_, dtype=bool)
            mask[self.selected_idx_] = True
            return mask

class SecondFilter:
    """
    SecondFilter supports:
      - L1-based feature selection
      - Tree-based feature selection
      - hsic_lasso feature selection

    This class adapts automatically for:
      - Binary classification
    """
    def __init__(self,
                 selection_method: str = "l1",
                 model_type: Optional[str] = "svc",
                 threshold: Union[str, float, int] = "mean",
                 random_state: int = 42,
                 **model_kwargs):
        """
        参数：
        selection_method: str
            "l1" , "tree" ,"hsic"三种选择方式，分别对应 L1,基于树模型,hsic的特征选择。

        model_type : str
            当 selection_method="l1" 时，可选："logistic", "svc"
            当 selection_method="tree" 时，可选："rf", "ada", "xgb"
            当 selection_method="hsic"时，此参数为 "classification" or "regression"

        threshold: Union[str, float]
            SelectFromModel 的特征重要性阈值，可设为 "mean", "median", 数字等。
            当为数字时，重要性>=该数值将被保留，其余丢弃，可填类似“1.25*mean”这样的字符串
            当selection_method="hsic"时，此参数只能为int

        random_state: int
            随机种子，确保结果可复现。

        model_kwargs: 其它传给基模型的参数
            比如 n_estimators、C、max_depth 等，可以在此传入。
            With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected
        """

        self.selection_method = selection_method.lower()
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        if self.selection_method=="l1" or self.selection_method=="tree":
            ####当l1和tree时，有base_estimator此私有属性
            self.base_estimator = self._build_base_estimator()  ##该私有方法只会在l1或者tree时会被调用
            self.selector = SelectFromModel(
                estimator=self.base_estimator,
                threshold=self.threshold
            )

        elif self.selection_method=="hsic":
            if self.model_type not in ("classification", "regression"):
                raise ValueError("For hsic, model_type must be 'classification' or 'regression'.")
            ####hsic时，无base_estimator此私有属性
            if not isinstance(self.threshold, int):
                raise ValueError("When selection_method='hsic', the threshold must be an integer "
                                  "specifying how many features to select, but got: "
                             f"{self.threshold} (type={type(self.threshold).__name__}).")

            self.selector = HSICLassoTransformer(
                mode=model_type,
                num_feat=self.threshold,
                **self.model_kwargs
            )
        else:
            raise ValueError("selection_method must be 'l1','tree','hsic'")

    def _build_base_estimator(self):
        if self.selection_method == "l1":
            if self.model_type == "logistic":
                estimator = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    random_state=self.random_state,
                    class_weight="balanced",
                    **self.model_kwargs
                )
            elif self.model_type == "svc":
                estimator = LinearSVC(
                    penalty="l1",
                    dual="auto",
                    loss="squared_hinge",
                    random_state=self.random_state,
                    class_weight="balanced",
                    **self.model_kwargs
                )
            else:
                raise ValueError(
                    "model_type must be 'logistic' or 'svc' when selection_method='l1'"
            )
        else:
            if self.model_type == "rf":
                estimator = RandomForestClassifier(
                    random_state=self.random_state,
                    **self.model_kwargs
                )
            elif self.model_type == "ada":
                estimator = AdaBoostClassifier(
                    random_state=self.random_state,
                    **self.model_kwargs
                )
            elif self.model_type == "xgb":
                estimator = XGBClassifier(
                    random_state=self.random_state,
                    eval_metric="logloss",
                    **self.model_kwargs
                )
            else:
                raise ValueError("model_type must be 'rf', 'ada', or 'xgb' when selection_method='tree'")

        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合并确定需要选择的特征。
        """
        self.selector.fit(X, y)
        selected_idx = self.get_support(indices=True)
        print(f"[SecondFilter] Total selected features: {len(selected_idx)}")
        return self

    def transform(self, X: np.ndarray)->np.ndarray:
        """
        使用已选出的特征，对新数据进行变换。
        """
        X_res=self.selector.transform(X)
        return X_res

    def fit_transform(self, X: np.ndarray, y: np.ndarray)->np.ndarray:
        """
        先拟合后变换。
        """
        X_new=self.selector.fit_transform(X, y)
        selected_idx = self.get_support(indices=True)
        print(f"[SecondFilter] Total selected features: {len(selected_idx)}")
        return X_new

    def get_support(self, indices: bool = False):
        """
        返回一个掩码或特征索引数组，指示哪些特征被选中了。
        """
        return self.selector.get_support(indices=indices)

    def get_feature_importances(self):
        """
        获取每个[输入特征]的重要性或系数：
            - 对于 L1-based (logistic / linearSVC)：返回模型系数
            - 对于 tree-based：返回模型的 feature_importances_
            - 对于 hsic: 直接返回模型的 feature_importances_
        """
        if self.selection_method == "l1" or self.selection_method== "tree":
            # 确保先 fit 再获取，l1和tree在拟合后selectfrommodel会有estimator_属性
            if not hasattr(self.selector, "estimator_"):
                raise AttributeError("SecondFilter has not been fit yet. Call `fit` first.")

            estimator_ = self.selector.estimator_

            # L1-based
            if self.selection_method == "l1":
                coefs = estimator_.coef_
                # LogisticRegression 通常 shape=(1, n_features)；LinearSVC 可能 shape=(n_classes, n_features)
                # 二分类这里一般是 (1, n_features) 或 (2, n_features) （对于'ovr'策略等）
                # 做一下 flatten
                if len(coefs.shape) > 1:
                    coefs = coefs.ravel()
                return coefs

            # Tree-based
            else:
                if not hasattr(estimator_, "feature_importances_"):
                    raise AttributeError(f"{type(estimator_).__name__} has no attribute 'feature_importances_'.")
                return estimator_.feature_importances_
        else:
            # hsic中，无有estimator_属性
            return self.selector.feature_importances_

    def set_params(self, **new_params):
        """
        动态更新参数。
        可根据需要修改此逻辑。
        """
        for key, val in new_params.items():
            setattr(self, key, val)

        if self.selection_method in ("l1", "tree"):
            self.base_estimator = self._build_base_estimator()
            self.selector = SelectFromModel(
                estimator=self.base_estimator,
                threshold=self.threshold
            )
        elif self.selection_method == "hsic":
            self.selector = HSICLassoTransformer(
                mode=self.model_type,
                num_feat=self.threshold,
                **self.model_kwargs
            )
        else:
            raise ValueError("selection_method must be 'l1','tree','hsic'")

        return self
