"""전처리부터 장비 단위 RUL 결정까지 연결하는 상위 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .automl import AutoMLTrajectoryEngine, CandidateScore
from .config import RULSystemConfig
from .preprocessing_and_onset import (
    FDCPreprocessor,
    OnsetDetectionResult,
    OnsetDetector,
    SensorControlLimits,
)


@dataclass(slots=True)
class SensorRULResult:
    sensor: str
    status: str
    onset_time: pd.Timestamp | None
    failure_time: pd.Timestamp | None
    rul_days: float | None
    rul_hours: float | None
    model_name: str | None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    formula: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    leaderboard: list[CandidateScore] = field(default_factory=list)
    fit_times: pd.DatetimeIndex | None = None
    fitted_raw: np.ndarray | None = None
    future_times: pd.DatetimeIndex | None = None
    future_raw: np.ndarray | None = None

    def to_dict(self, limit: SensorControlLimits) -> dict[str, Any]:
        return {
            "센서": self.sensor,
            "상태": self.status,
            "열화_시작시각": self.onset_time.isoformat() if self.onset_time is not None else None,
            "예상_고장시각": self.failure_time.isoformat() if self.failure_time is not None else None,
            "잔여수명_일": self.rul_days,
            "잔여수명_시간": self.rul_hours,
            "모델": self.model_name,
            "모델_파라미터": self.model_parameters,
            "모델식": self.formula,
            "평가지표": self.metrics,
            "기준_평균": limit.baseline_mean,
            "기준_표준편차": limit.baseline_std,
            "열화_방향": limit.direction_label,
            "2시그마_경고한계": limit.warning_threshold,
            "3시그마_고장한계": limit.failure_threshold,
            "AutoML_순위표": [score.to_dict() for score in self.leaderboard],
        }


@dataclass(slots=True)
class EquipmentRULResult:
    status: str
    analysis_time: pd.Timestamp
    estimated_failure_time: pd.Timestamp | None
    rul_days: float | None
    rul_hours: float | None
    critical_sensor: str | None
    processed_data: pd.DataFrame
    timestamp_column: str
    sensor_columns: list[str]
    detection: OnsetDetectionResult
    sensor_results: dict[str, SensorRULResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_failure_date(self) -> str | None:
        if self.estimated_failure_time is None:
            return None
        return self.estimated_failure_time.strftime("%Y-%m-%d")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "시스템_상태": self.status,
            "분석_기준시각": self.analysis_time.isoformat(),
            "예상_고장일": self.estimated_failure_date,
            "예상_고장시각": (
                self.estimated_failure_time.isoformat()
                if self.estimated_failure_time is not None
                else None
            ),
            "잔여수명_RUL_일": self.rul_days,
            "잔여수명_RUL_시간": self.rul_hours,
            "주요_위험센서": self.critical_sensor,
            "탐지된_열화센서": self.detection.active_sensors,
            "전역_열화_시작시각": (
                self.detection.onset_time.isoformat()
                if self.detection.onset_time is not None
                else None
            ),
            "주의사항": self.warnings,
            "센서별_결과": {
                sensor: result.to_dict(self.detection.sensor_limits[sensor])
                for sensor, result in self.sensor_results.items()
            },
        }


class RULPredictor:
    """일반화된 다중 센서 FDC RUL 예측기."""

    def __init__(self, config: RULSystemConfig | None = None):
        self.config = config or RULSystemConfig()
        self.config.validate()
        self.preprocessor = FDCPreprocessor(self.config.preprocessing)
        self.onset_detector = OnsetDetector(self.config.onset)
        self.automl = AutoMLTrajectoryEngine(self.config.automl)

    def predict(self, data: pd.DataFrame) -> EquipmentRULResult:
        processed = self.preprocessor.fit_transform(data)
        sensors = list(self.preprocessor.sensor_columns_)
        timestamp_column = self.config.preprocessing.timestamp_column
        detection = self.onset_detector.detect(processed, sensors, timestamp_column)
        analysis_time = pd.Timestamp(processed[timestamp_column].iloc[-1])
        sensor_results: dict[str, SensorRULResult] = {}
        warnings: list[str] = []

        for sensor in sensors:
            limit = detection.sensor_limits[sensor]
            if sensor not in detection.active_sensors or limit.onset_index is None:
                sensor_results[sensor] = SensorRULResult(
                    sensor=sensor,
                    status="드리프트 없음",
                    onset_time=None,
                    failure_time=None,
                    rul_days=None,
                    rul_hours=None,
                    model_name=None,
                )
                continue

            if limit.first_failure_index is not None:
                failure_time = pd.Timestamp(
                    processed[timestamp_column].iloc[limit.first_failure_index]
                )
                status = "급격 고장" if detection.catastrophic_sensor == sensor else "고장 한계 도달"
                sensor_results[sensor] = SensorRULResult(
                    sensor=sensor,
                    status=status,
                    onset_time=limit.onset_time,
                    failure_time=failure_time,
                    rul_days=0.0,
                    rul_hours=0.0,
                    model_name=None,
                )
                continue

            segment = processed.iloc[limit.onset_index :].copy()
            if len(segment) < self.config.onset.min_active_points:
                warning = (
                    f"센서 '{sensor}'는 열화 구간이 {len(segment)}개로 짧아 "
                    "신뢰 가능한 외삽을 생략했습니다."
                )
                warnings.append(warning)
                sensor_results[sensor] = SensorRULResult(
                    sensor=sensor,
                    status="열화 추세 데이터 부족",
                    onset_time=limit.onset_time,
                    failure_time=None,
                    rul_days=None,
                    rul_hours=None,
                    model_name=None,
                )
                continue

            segment_times = pd.to_datetime(segment[timestamp_column])
            x_days = (segment_times - segment_times.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
            y_raw = segment[f"{sensor}__smooth"].to_numpy(dtype=float)
            try:
                model_result = self.automl.fit(
                    x_days,
                    y_raw,
                    limit.direction,
                    failure_threshold_raw=limit.failure_threshold,
                )
                extrapolation = self.automl.extrapolate(
                    model_result,
                    last_x_days=float(x_days[-1]),
                    threshold_raw=limit.failure_threshold,
                    direction=limit.direction,
                )
            except (ValueError, RuntimeError) as exc:
                warnings.append(f"센서 '{sensor}' 모델링 실패: {exc}")
                sensor_results[sensor] = SensorRULResult(
                    sensor=sensor,
                    status="모델링 실패",
                    onset_time=limit.onset_time,
                    failure_time=None,
                    rul_days=None,
                    rul_hours=None,
                    model_name=None,
                )
                continue

            fitted = np.asarray(model_result.estimator.predict(x_days.reshape(-1, 1))).reshape(-1)
            future_times = pd.DatetimeIndex(
                segment_times.iloc[0] + pd.to_timedelta(extrapolation.x_days, unit="D")
            )
            if extrapolation.crossing_x_days is None:
                failure_time = None
                rul_days = None
                rul_hours = None
                status = "외삽 범위 내 고장 교차 없음"
                warnings.append(
                    f"센서 '{sensor}'는 {self.config.automl.max_extrapolation_days:g}일 "
                    "외삽 범위 안에서 3σ 한계와 교차하지 않았습니다."
                )
            else:
                failure_time = pd.Timestamp(
                    segment_times.iloc[0]
                    + pd.to_timedelta(extrapolation.crossing_x_days, unit="D")
                )
                rul_hours = max(
                    float((failure_time - analysis_time).total_seconds() / 3600.0), 0.0
                )
                rul_days = rul_hours / 24.0
                status = "고장 시점 예측 완료"

            metrics = {
                "교차검증_RMSE": model_result.best_score.cv_rmse,
                "교차검증_R2": model_result.best_score.cv_r2,
                "학습_RMSE": model_result.best_score.train_rmse,
                "학습_R2": model_result.best_score.train_r2,
                "AIC": model_result.best_score.aic,
                "BIC": model_result.best_score.bic,
            }
            sensor_results[sensor] = SensorRULResult(
                sensor=sensor,
                status=status,
                onset_time=limit.onset_time,
                failure_time=failure_time,
                rul_days=rul_days,
                rul_hours=rul_hours,
                model_name=model_result.model_name,
                model_parameters=model_result.parameters,
                formula=model_result.formula,
                metrics=metrics,
                leaderboard=model_result.leaderboard,
                fit_times=pd.DatetimeIndex(segment_times),
                fitted_raw=fitted,
                future_times=future_times,
                future_raw=extrapolation.predicted_raw,
            )

        failed = [
            result
            for result in sensor_results.values()
            if result.status in {"급격 고장", "고장 한계 도달"}
        ]
        predicted = [
            result
            for result in sensor_results.values()
            if result.failure_time is not None
            and result.status not in {"급격 고장", "고장 한계 도달"}
        ]
        if failed:
            critical = min(failed, key=lambda item: item.failure_time or analysis_time)
            status = critical.status
            failure_time = critical.failure_time
            rul_days = 0.0
            rul_hours = 0.0
        elif predicted:
            critical = min(predicted, key=lambda item: item.failure_time or pd.Timestamp.max)
            status = "활성 열화/고장 시점 예측 완료"
            failure_time = critical.failure_time
            rul_days = critical.rul_days
            rul_hours = critical.rul_hours
        elif detection.active_sensors:
            critical = None
            status = "활성 열화/예측 불확정"
            failure_time = None
            rul_days = None
            rul_hours = None
        else:
            critical = None
            status = "정상/드리프트 없음"
            failure_time = None
            rul_days = None
            rul_hours = None

        return EquipmentRULResult(
            status=status,
            analysis_time=analysis_time,
            estimated_failure_time=failure_time,
            rul_days=rul_days,
            rul_hours=rul_hours,
            critical_sensor=critical.sensor if critical is not None else None,
            processed_data=processed,
            timestamp_column=timestamp_column,
            sensor_columns=sensors,
            detection=detection,
            sensor_results=sensor_results,
            warnings=warnings,
        )
