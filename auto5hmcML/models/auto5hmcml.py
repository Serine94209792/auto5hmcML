from __future__ import annotations

import os.path

from .trainer import (
    trainmodels as _trainmodels,
    ensemblemodels as _ensemblemodels,
    stackmodels as _stackmodels,
)
import inspect, functools
from typing import Any,Optional,Callable,TypeVar,Union,List
import pandas as pd
F = TypeVar("F", bound=Callable[..., Any])

def copy_sig(src: Callable[..., Any]) -> Callable[[F], F]:
    """
    @copy_sig(_trainmodels)
    """
    def decorator(dst: F) -> F:
        dst.__signature__ = inspect.signature(src)
        functools.update_wrapper(dst, src,
                                 assigned=('__doc__', '__name__'))
        return dst
    return decorator

class Auto5hmcML:
    def __init__(self, *,
                 ensemble: bool = False,
                 stack: bool = False,
                 path: Optional[str]=None) -> None:
        """
        Parameters
        ----------
        path : str, default None, path to where your trainset and testset saved
        ensemble : bool, default False
        stack : bool, default False
        you must have ensemble=True firstly then stack=True. stack cannot be True when ensemble = False
        """
        self.ensemble = ensemble
        self.stack = stack
        self.path = path

    @staticmethod
    def _filter_kwargs(func, kwargs: dict[str, Any]) -> dict[str, Any]:
        """pop无关参数 ，避免传递给不接受它们的函数"""
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @copy_sig(_trainmodels)
    def trainmodels(self, **kwargs: Any) -> None:
        usable = self._filter_kwargs(_trainmodels, kwargs)
        return _trainmodels(**usable)

    @copy_sig(_ensemblemodels)
    def ensemblemodels(self, **kwargs: Any) -> None:
        usable = self._filter_kwargs(_ensemblemodels, kwargs)
        return _ensemblemodels(**usable)

    @copy_sig(_stackmodels)
    def stackmodels(self, **kwargs: Any) -> None:
        usable = self._filter_kwargs(_stackmodels, kwargs)
        return _stackmodels(**usable)

    def train(
            self,
            *,
            # ---- trainmodels核心参数 ----
            trainset: pd.DataFrame,
            testset: pd.DataFrame,
            y: str,
            drop_y: Optional[List[str]] = None,
            scoring: Union[str, callable] = "accuracy",
            cv: int = 10,
            n_trials: int = 100,
            direction: str = "maximize",
            scaler: bool = True,
            imbalencer: bool = True,
            first_filter: bool = True,
            second_filter: bool = True,
            learner: bool = True,
            n_components_hp: bool = True,
            n_components: Optional[int] = None,
            max_n_compoents: int = 10,
            imbalencer_type: Optional[str] = "tomeklinks",
            selection_method: Optional[str] = "l1",
            filter_type: Optional[str] = "svc",
            learner_type: Optional[str] = "pca",
            imbalencer_kwargs: dict | None = None,
            ff_kwargs: dict | None = None,
            sf_kwargs: dict | None = None,
            lr_kwargs: dict | None = None,
            # ---- Voting / Stacking 额外参数 ----
            top_n_ensemble: int = 5,
            top_n_stack: int = 20,
            add_to_original: bool = False,
            n_jobs: int = -1,   ###cv时使用线程
    ) -> None:
        """
        Parameters
        ----------
        params：训练集测试集，y预测标签，drop_y不需要的标签，预测标签需要在测试机和训练集中
        scoring为交叉验证中的评估参数，默认acc，也可以为自定义scoring
        cv交叉验证折数
        n_trials超参数寻找次数
        direction：优化方向
        当leaner为Ture时，若n_components_hp为True，则n_components为超参数可不提供值，若n_components_hp为False，则n_components必须提供值
        max_n_compoents默认为10，作为超参数的最大搜索值，推荐为训练集样本数的1/10
        kwargs继承自createpipeline
        top_n_ensemble : int, default 5
            *仅当* ``self.ensemble=True`` 时启用；传给
            :py:func:`trainer.ensemblemodels` 的 ``top_n``
        top_n_stack : int, default 20
            *仅当* ``self.stack=True`` 时启用；传给
            :py:func:`trainer.stackmodels` 的 ``top_n``
        add_to_original : bool, default False
            传给 ``stackmodels`` 的 ``add_to_original``
        """

        if self.path is not None:
            if not os.path.exists(self.path):
                raise FileNotFoundError("The path does not exist.")
            else:
                os.chdir(self.path)

        self.trainmodels(
            trainset=trainset,
            testset=testset,
            y=y,
            drop_y=drop_y,
            scoring=scoring,
            cv=cv,
            n_trials=n_trials,
            direction=direction,
            scaler=scaler,
            imbalencer=imbalencer,
            first_filter=first_filter,
            second_filter=second_filter,
            learner=learner,
            n_components_hp=n_components_hp,
            n_components=n_components,
            max_n_compoents=max_n_compoents,
            imbalencer_type=imbalencer_type,
            selection_method=selection_method,
            filter_type=filter_type,
            learner_type=learner_type,
            imbalencer_kwargs=imbalencer_kwargs,
            ff_kwargs=ff_kwargs,
            sf_kwargs=sf_kwargs,
            lr_kwargs=lr_kwargs,
        )

        if self.ensemble:
            self.ensemblemodels(
                trainset=trainset,
                testset=testset,
                y=y,
                drop_y=drop_y,
                scoring=scoring,
                cv=cv,
                n_trials=n_trials,
                direction=direction,
                top_n=top_n_ensemble,
                n_jobs=n_jobs,
            )

        if self.stack:
            self.stackmodels(
                trainset=trainset,
                testset=testset,
                y=y,
                drop_y=drop_y,
                scoring=scoring,
                cv=cv,
                n_trials=n_trials,
                direction=direction,
                top_n=top_n_stack,
                add_to_original=add_to_original,
            )

