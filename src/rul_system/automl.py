"""시계열 교차 검증 기반 AutoML과 고장 한계 외삽."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

from .config import AutoMLConfig
from .models import build_model_candidates, estimator_formula


@dataclass(slots=True)
class CandidateScore:
    model_name: str
    parameters: dict[str, Any]
    cv_rmse: float
    cv_r2: float
    train_rmse: float
    train_r2: float
    aic: float
    bic: float
    trend_penalty: float
    selection_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "모델": self.model_name,
            "파라미터": self.parameters,
            "교차검증_RMSE": self.cv_rmse,
            "교차검증_R2": self.cv_r2,
            "학습_RMSE": self.train_rmse,
            "학습_R2": self.train_r2,
            "AIC": self.aic,
            "BIC": self.bic,
            "추세_페널티": self.trend_penalty,
            "선택_점수": self.selection_score,
        }


@dataclass(slots=True)
class AutoMLResult:
    model_name: str
    estimator: BaseEstimator
    parameters: dict[str, Any]
    formula: str
    best_score: CandidateScore
    leaderboard: list[CandidateScore]


@dataclass(slots=True)
class ExtrapolationResult:
    x_days: np.ndarray
    predicted_raw: np.ndarray
    crossing_x_days: float | None


def _parameter_count(estimator: BaseEstimator, model_name: str) -> int:
    if model_name == "선형 회귀":
        return 2
    if model_name == "다항 회귀":
        degree = int(estimator.named_steps["poly"].degree)
        return degree + 1
    if model_name == "지수 성장":
        return 3
    params = estimator.get_params(deep=True)
    return max(2, sum(isinstance(value, (int, float)) for value in params.values()))


def _information_criteria(y: np.ndarray, predicted: np.ndarray, parameters: int) -> tuple[float, float]:
    n = len(y)
    rss = max(float(np.sum((y - predicted) ** 2)), np.finfo(float).tiny)
    aic = n * np.log(rss / n) + 2 * parameters
    bic = n * np.log(rss / n) + parameters * np.log(n)
    return float(aic), float(bic)


class AutoMLTrajectoryEngine:
    """GridSearch와 확장 시계열 검증으로 열화 궤적 모델을 선택한다."""

    def __init__(self, config: AutoMLConfig | None = None):
        self.config = config or AutoMLConfig()

    def fit(
        self,
        x_days: np.ndarray,
        y_raw: np.ndarray,
        direction: int,
        failure_threshold_raw: float | None = None,
    ) -> AutoMLResult:
        self.config.validate()
        x = np.asarray(x_days, dtype=float).reshape(-1, 1)
        y = np.asarray(y_raw, dtype=float).reshape(-1)
        finite = np.isfinite(x[:, 0]) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(y) < 6:
            raise ValueError("AutoML 모델링에는 유효한 열화 지점이 최소 6개 필요합니다.")
        if np.ptp(x[:, 0]) <= 0:
            raise ValueError("모델링 시간축의 범위가 0입니다.")

        candidates = [
            candidate
            for candidate in build_model_candidates(
                random_state=self.config.random_state,
                include_xgboost=self.config.enable_xgboost_if_available,
            )
            if candidate.name in self.config.enabled_models
        ]
        if not candidates:
            raise ValueError("활성화된 AutoML 모델이 없습니다.")

        max_splits = max(2, min(self.config.cv_splits, len(y) - 2))
        splitter = TimeSeriesSplit(n_splits=max_splits)
        y_scale = max(float(np.std(y, ddof=1)), float(np.ptp(y)) * 0.1, 1e-9)
        scores: list[tuple[CandidateScore, BaseEstimator]] = []

        for candidate in candidates:
            for parameters in ParameterGrid(candidate.parameter_grid):
                fold_rmse: list[float] = []
                fold_r2: list[float] = []
                failed = False
                for train_index, validation_index in splitter.split(x):
                    if len(train_index) < 4:
                        continue
                    model = clone(candidate.estimator).set_params(**parameters)
                    try:
                        model.fit(x[train_index], y[train_index])
                        predicted = np.asarray(model.predict(x[validation_index])).reshape(-1)
                    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                        failed = True
                        break
                    fold_rmse.append(float(mean_squared_error(y[validation_index], predicted) ** 0.5))
                    if len(validation_index) > 1:
                        fold_r2.append(float(r2_score(y[validation_index], predicted)))
                if failed or not fold_rmse:
                    continue

                final_model = clone(candidate.estimator).set_params(**parameters)
                try:
                    final_model.fit(x, y)
                    train_prediction = np.asarray(final_model.predict(x)).reshape(-1)
                    trend_penalty = self._trend_penalty(
                        final_model,
                        x[:, 0],
                        y,
                        direction,
                        failure_threshold_raw,
                        self.config.max_extrapolation_days,
                    )
                except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                    continue

                train_rmse = float(mean_squared_error(y, train_prediction) ** 0.5)
                train_r2 = float(r2_score(y, train_prediction))
                cv_rmse = float(np.mean(fold_rmse))
                cv_r2 = float(np.mean(fold_r2)) if fold_r2 else -1.0
                parameter_count = _parameter_count(final_model, candidate.name)
                aic, bic = _information_criteria(y, train_prediction, parameter_count)
                r2_penalty = 0.08 * (1.0 - float(np.clip(cv_r2, -1.0, 1.0)))
                complexity_penalty = 0.002 * parameter_count
                selection_score = (
                    cv_rmse / y_scale
                    + r2_penalty
                    + self.config.trend_penalty_weight * trend_penalty
                    + complexity_penalty
                )
                score = CandidateScore(
                    model_name=candidate.name,
                    parameters=parameters,
                    cv_rmse=cv_rmse,
                    cv_r2=cv_r2,
                    train_rmse=train_rmse,
                    train_r2=train_r2,
                    aic=aic,
                    bic=bic,
                    trend_penalty=trend_penalty,
                    selection_score=float(selection_score),
                )
                scores.append((score, final_model))

        if not scores:
            raise RuntimeError("모든 AutoML 후보 모델의 학습이 실패했습니다.")
        scores.sort(key=lambda item: item[0].selection_score)
        best_score, best_model = scores[0]
        return AutoMLResult(
            model_name=best_score.model_name,
            estimator=best_model,
            parameters=best_score.parameters,
            formula=estimator_formula(best_score.model_name, best_model),
            best_score=best_score,
            leaderboard=[item[0] for item in scores],
        )

    @staticmethod
    def _trend_penalty(
        estimator: BaseEstimator,
        x: np.ndarray,
        y: np.ndarray,
        direction: int,
        failure_threshold_raw: float | None,
        max_extrapolation_days: float,
    ) -> float:
        span = max(float(np.ptp(x)), 1e-9)
        evaluation_horizon = min(max_extrapolation_days, max(span * 5.0, 1.0))
        future = np.linspace(
            float(x[-1]),
            float(x[-1] + evaluation_horizon),
            128,
        ).reshape(-1, 1)
        predicted = np.asarray(estimator.predict(future)).reshape(-1)
        local_count = max(3, int(len(predicted) * min(0.25 * span / evaluation_horizon, 0.25)))
        progress = direction * float(predicted[local_count - 1] - predicted[0])
        observed_progress = max(abs(float(y[-1] - y[0])), float(np.ptp(y)) * 0.1, 1e-9)
        if not np.all(np.isfinite(predicted)):
            return 4.0
        threshold_penalty = 0.0
        if failure_threshold_raw is not None:
            reaches_failure = np.any(
                direction * predicted >= direction * float(failure_threshold_raw)
            )
            if not reaches_failure:
                threshold_penalty = 1.5
        if progress < -0.01 * observed_progress:
            return 2.0 + threshold_penalty
        if progress < 0.02 * observed_progress:
            return 1.0 + threshold_penalty
        local_steps = direction * np.diff(predicted[:local_count])
        reversal_fraction = float(np.mean(local_steps < -0.01 * observed_progress))
        return min(1.0, reversal_fraction) + threshold_penalty

    def extrapolate(
        self,
        result: AutoMLResult,
        last_x_days: float,
        threshold_raw: float,
        direction: int,
    ) -> ExtrapolationResult:
        """예측 곡선과 3σ 고장 한계의 최초 교차점을 계산한다."""

        x_future = np.linspace(
            float(last_x_days),
            float(last_x_days + self.config.max_extrapolation_days),
            self.config.extrapolation_grid_points,
        )
        predicted = np.asarray(result.estimator.predict(x_future.reshape(-1, 1))).reshape(-1)
        reached = direction * predicted >= direction * threshold_raw
        matches = np.flatnonzero(reached & np.isfinite(predicted))
        if not len(matches):
            return ExtrapolationResult(x_future, predicted, None)

        index = int(matches[0])
        if index == 0:
            crossing = float(x_future[0])
        else:
            x0, x1 = float(x_future[index - 1]), float(x_future[index])
            y0, y1 = float(predicted[index - 1]), float(predicted[index])
            if np.isclose(y1, y0):
                crossing = x1
            else:
                fraction = float(np.clip((threshold_raw - y0) / (y1 - y0), 0.0, 1.0))
                crossing = x0 + fraction * (x1 - x0)
        kept_x = np.append(x_future[:index], crossing)
        kept_y = np.append(predicted[:index], threshold_raw)
        return ExtrapolationResult(kept_x, kept_y, crossing)
