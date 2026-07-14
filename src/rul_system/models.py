"""열화 궤적에 사용할 회귀 모델 라이브러리."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


class ExponentialGrowthRegressor(RegressorMixin, BaseEstimator):
    """y = a·exp(b·x_scaled) + c 형태의 양방향 지수 회귀."""

    def __init__(self, maxfev: int = 20000):
        self.maxfev = maxfev

    @staticmethod
    def _curve(x: np.ndarray, amplitude: float, rate: float, offset: float) -> np.ndarray:
        return amplitude * np.exp(np.clip(rate * x, -60.0, 60.0)) + offset

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExponentialGrowthRegressor":
        x = np.asarray(X, dtype=float).reshape(-1)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(x) < 4 or np.ptp(x) <= 0:
            raise ValueError("지수 회귀에는 서로 다른 X 값 4개 이상이 필요합니다.")
        self.x_min_ = float(x.min())
        self.x_scale_ = max(float(np.ptp(x)), 1e-12)
        scaled = (x - self.x_min_) / self.x_scale_
        delta = float(target[-1] - target[0])
        amplitude = delta if abs(delta) > 1e-12 else max(float(np.std(target)), 1e-6)
        initial = [amplitude, 1.0, float(target[0] - amplitude)]
        target_span = max(float(np.ptp(target)), abs(float(np.mean(target))) * 1e-6, 1e-6)
        bounds = (
            [-1000 * target_span, -10.0, float(target.min() - 1000 * target_span)],
            [1000 * target_span, 10.0, float(target.max() + 1000 * target_span)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            self.amplitude_, self.rate_, self.offset_ = curve_fit(
                self._curve,
                scaled,
                target,
                p0=initial,
                bounds=bounds,
                maxfev=self.maxfev,
            )[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "amplitude_"):
            raise RuntimeError("predict 전에 fit을 호출해야 합니다.")
        x = np.asarray(X, dtype=float).reshape(-1)
        scaled = (x - self.x_min_) / self.x_scale_
        return self._curve(scaled, self.amplitude_, self.rate_, self.offset_)


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    name: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]


def build_model_candidates(random_state: int = 42, include_xgboost: bool = True) -> list[ModelCandidate]:
    """설치 환경에 맞는 모델 및 GridSearch 후보를 반환한다."""

    candidates = [
        ModelCandidate(
            name="선형 회귀",
            estimator=Pipeline([("ridge", Ridge())]),
            parameter_grid={"ridge__alpha": [0.0, 0.01, 0.1, 1.0]},
        ),
        ModelCandidate(
            name="다항 회귀",
            estimator=Pipeline(
                [
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ("ridge", Ridge()),
                ]
            ),
            parameter_grid={
                "poly__degree": [2, 3],
                "ridge__alpha": [0.01, 0.1, 1.0],
            },
        ),
        ModelCandidate(
            name="지수 성장",
            estimator=ExponentialGrowthRegressor(),
            parameter_grid={"maxfev": [10000, 20000]},
        ),
        ModelCandidate(
            name="SVR",
            estimator=TransformedTargetRegressor(
                regressor=Pipeline([("scale", StandardScaler()), ("svr", SVR())]),
                transformer=StandardScaler(),
            ),
            parameter_grid={
                "regressor__svr__C": [1.0, 10.0, 100.0],
                "regressor__svr__gamma": ["scale", 0.1],
                "regressor__svr__epsilon": [0.01, 0.1],
            },
        ),
        ModelCandidate(
            name="Gradient Boosting",
            estimator=GradientBoostingRegressor(random_state=random_state, loss="huber"),
            parameter_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
            },
        ),
    ]

    if include_xgboost:
        try:
            from xgboost import XGBRegressor

            candidates.append(
                ModelCandidate(
                    name="XGBoost",
                    estimator=XGBRegressor(
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                    parameter_grid={
                        "n_estimators": [100, 250],
                        "learning_rate": [0.03, 0.1],
                        "max_depth": [2, 4],
                    },
                )
            )
        except ImportError:
            pass
    return candidates


def estimator_formula(name: str, estimator: BaseEstimator) -> str:
    """그림 내부에 넣을 간결한 모델 수식을 만든다."""

    if name == "지수 성장":
        model = estimator
        return (
            f"y = {model.amplitude_:.4g}·exp({model.rate_:.4g}·xₛ) "
            f"+ {model.offset_:.4g}"
        )
    if name == "선형 회귀":
        ridge = estimator.named_steps["ridge"]
        return f"y = {float(ridge.coef_[0]):.4g}·x + {float(ridge.intercept_):.4g}"
    if name == "다항 회귀":
        ridge = estimator.named_steps["ridge"]
        terms = [f"{float(ridge.intercept_):.4g}"]
        for power, coefficient in enumerate(np.asarray(ridge.coef_).reshape(-1), start=1):
            terms.append(f"{float(coefficient):+.4g}·x^{power}")
        return "y = " + " ".join(terms)
    if name == "SVR":
        return "y = SVR(x; C, γ, ε)"
    if name == "XGBoost":
        return "y = Σ η·fₖ(x)  (XGBoost)"
    return "y = Σ η·hₖ(x)  (Gradient Boosting)"
