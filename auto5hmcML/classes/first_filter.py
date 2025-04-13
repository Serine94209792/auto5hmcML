from typing import Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import GenericUnivariateSelect
from scipy.stats import pearsonr, spearmanr, kruskal
from dcor import distance_correlation
import warnings
from sklearn import feature_selection


class CustomUnivariateSelect(BaseEstimator, TransformerMixin):
    """
        一个自定义的单因素特征选择类，类似 sklearn 的 GenericUnivariateSelect。

        支持以下统计方法：
        1. Pearson 相关系数 (method='pearson')， 适用于两个连续变量
        2. Spearman 相关系数 (method='spearman')， 适用于两个连续变量
        3. 距离相关系数 Distance Correlation (method='distancecorr')， 适用于两个连续变量
        4. Kruskal 检验 (method='kruskal')， 适用于一个连续变量，一个分类变量

        mode可以为k_best, percentile, pvalue
        参数
        ----------
        method : str, 可选，默认 'pearson'
            指定使用哪种单因素统计方法，可取值:
            ['pearson', 'spearman', 'distancecorr', 'kruskal'].

        mode : str, 可选，默认 'percentile'
            特征选择模式。k_best, percentile, pvalue

        param : int, 可选，默认 1
            对于 'k_best' 模式，保留的特征数目。
            对于 'percentile' 模式，表示前 param 百分位（0 ~ 100）。
            对于 'pvalue' 模式，p 值的阈值。

        属性
        ----------
        scores_ : ndarray, shape (n_features,)
            每个特征的得分（不同方法的得分含义不同）。

        pvalues_ : ndarray, shape (n_features,)
            每个特征对应的 p 值（对于 DistanceCorrelation 不计算 p 值，返回 np.nan）。

        support_ : ndarray, shape (n_features,), dtype bool
            用于指示哪些特征被选中（True 为被选中）。
        """
    def __init__(self,
                 score_func: str= "spearman",
                 mode: str = "percentile",
                 param: Union[int, float] = 1):
        self.score_func = score_func
        self.mode = mode
        self.param = param
        self.scores_ = None
        self.pvalues_ = None
        self.support_ = None
        self.n_features_in_ = None
        self.n_features_out_= None

    def _score_pearson(self, x:np.ndarray, y:np.ndarray):
        """
        使用 Pearson 相关系数对单个特征打分。
        返回: (score, p_value)
        这里默认以绝对相关系数作为打分值。
        """
        if np.all(x == x[0]):
            warnings.warn(
                "Warning: x is constant, returning 0.0, 1.0"
            )
            return 0.0, 1.0
        score, p = pearsonr(x, y)
        return abs(score), p

    def _score_spearman(self, x:np.ndarray, y:np.ndarray):
        if np.all(x == x[0]):
            warnings.warn(
                "Warning: x is constant, returning 0.0, 1.0"
            )
            return 0.0, 1.0
        score, p = spearmanr(x, y)
        return abs(score), p

    def _score_kruskal(self, x:np.ndarray, y:np.ndarray):
        """
        使用 Kruskal 检验对单个特征进行打分。
        一般用于分类任务：假设 y 为类别型数据。
        将 x 根据 y 的不同类别进行分组，然后做 Kruskal 检验。
        返回: (H_stat, p_value)
        这里默认以检验统计量 H_stat 作为打分值。
        """
        if np.all(x == x[0]):
            warnings.warn(
                "Warning: x is constant, returning 0.0, 1.0"
            )
            return 0.0, 1.0

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            warnings.warn(
                "Warning: y has less than 2 unique classes, returning 0.0, 1.0"
            )
            return 0.0, 1.0

        groups = [x[y == cls] for cls in unique_classes]

        stat, p = kruskal(*groups)
        return stat, p

    def _score_distancecorr(self, x:np.ndarray, y:np.ndarray):
        """
        使用dcor包中的distance_correlation
        返回: (distance_corr, p_value)
        其中p_value=np.nan
        """
        x=x.astype(float)
        y=y.astype(float)
        ####使用float可以加速
        distance_corr = distance_correlation(x, y)
        return distance_corr, np.nan

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        if self.score_func == 'pearson':
            scoring_func = self._score_pearson
        elif self.score_func == 'spearman':
            scoring_func = self._score_spearman
        elif self.score_func == 'distancecorr':
            scoring_func = self._score_distancecorr
        elif self.score_func == 'kruskal':
            scoring_func = self._score_kruskal
        else:
            raise ValueError(f"score_func {self.score_func} 不在可用范围内。")

        scores = []
        pvalues = []

        for i in range(n_features):
            score, pvalue = scoring_func(X[:, i], y)
            scores.append(score)
            pvalues.append(pvalue)

        self.scores_ = np.array(scores)
        self.pvalues_ = np.array(pvalues)

        if self.mode == 'k_best':
            if not isinstance(self.param, int) or self.param <= 0:
                raise ValueError("当 mode='k_best' 时，param 必须是 > 0 的整数")
            indices = np.argsort(self.scores_)[::-1]
            top_k_idx = indices[:self.param]
            support = np.zeros(n_features, dtype=bool)
            support[top_k_idx] = True
            self.n_features_out_ = self.param
            self.support_ = support

        elif self.mode == "percentile":
            if not (0 <= self.param <= 100):
                raise ValueError("当 mode='percentile' 时，param 必须在 [0, 100] 范围内")
            threshold = np.percentile(self.scores_, self.param)
            support = self.scores_ >= threshold
            self.n_features_out_ = np.sum(support)
            self.support_ = support

        elif self.mode == "pvalue":
            if self.score_func == "distancecorr":
                raise ValueError("DistanceCorrelation 不支持 pvalue 模式。")
            threshold = self.param
            support = self.pvalues_ <= threshold
            self.n_features_out_ = np.sum(support)
            self.support_ = support

        else:
            raise ValueError(f"mode {self.mode} 暂不支持。")

        return self

    def transform(self, X: np.ndarray):
        if self.support_ is None:
            raise ValueError("请先调用 fit")
        return X[:, self.support_]

    def fit_transfrom(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False):
        if self.support_ is None:
            raise ValueError("请先调用 fit")
        if indices:
            return np.where(self.support_==True)[0]
        else:
            return self.support_

    def set_param(self, param_name:str, param_value:any):
        """
        动态设置某个参数（如 score_func、mode、param 等）。
        设置后如果需要生效，需要重新调用 fit/fit_transform。

        参数
        ----------
        param_name : str
            要修改的参数名称。
        param_value : any
            要修改的参数值。

        返回
        ----------
        self : object
        """
        if not hasattr(self, param_name):
            raise ValueError(f"{self.__class__.__name__!r} 没有名为 {param_name!r} 的参数.")
        setattr(self, param_name, param_value)
        return self

def FirstFilter(mode: str="percentile",
                function: str="mutual_info_classif",
                param: Union[int,float]=1) -> Union[GenericUnivariateSelect,CustomUnivariateSelect]:
    """
    使用单因素筛选特征,可用sklearn的GenericUnivariateSelect和自定义的CustomUnivariateSelect

    params：
    mode默认为percentile
    function默认为mutual_info_classif，支持['pearson', 'spearman', 'distancecorr', 'kruskal']和GenericUnivariateSelect里所有的
    param默认为percentile 1（指前1%）

    return:
    GenericUnivariateSelect类或CustomUnivariateSelect类

    注意互信息和dcor不支持p_val！！！！
    """
    if function not in ['pearson', 'spearman', 'distancecorr', 'kruskal']:
        first_filter = GenericUnivariateSelect(
            score_func=getattr(feature_selection,function),
            mode=mode,
            param=param
        )
    else:
        first_filter = CustomUnivariateSelect(
            score_func=function,
            mode=mode,
            param=param
        )
    return first_filter

