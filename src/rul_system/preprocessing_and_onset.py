"""FDC 전처리, 노이즈 제거, 특징 생성, 열화 시작점 탐지.

표시용 센서 열은 원시 단위를 유지한다. 결측 보간과 정규화는 각각
``__clean`` 및 ``__zscore`` 파생 열에서만 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import OnsetConfig, PreprocessingConfig


DERIVED_SUFFIXES = (
    "__clean",
    "__smooth",
    "__rolling_mean",
    "__rolling_std",
    "__gradient",
    "__cum_degradation",
    "__zscore",
)


def load_fdc_data(path: str | Path) -> pd.DataFrame:
    """CSV 또는 Parquet FDC 파일을 읽는다."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet 입력 엔진이 없습니다. 'pip install -e .[parquet]'로 설치하세요."
            ) from exc
    raise ValueError("지원 형식은 CSV(.csv)와 Parquet(.parquet)입니다.")


def _infer_sensor_columns(frame: pd.DataFrame, timestamp_column: str) -> list[str]:
    sensors: list[str] = []
    for column in frame.columns:
        if column == timestamp_column or any(column.endswith(s) for s in DERIVED_SUFFIXES):
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(frame) * 0.5)):
            sensors.append(column)
    return sensors


class FDCPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn 호환 FDC 시계열 전처리기."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def fit(self, X: pd.DataFrame, y: object = None) -> "FDCPreprocessor":
        del y
        self.config.validate()
        if self.config.timestamp_column not in X.columns:
            raise ValueError(
                f"시간 열 '{self.config.timestamp_column}'이 입력 데이터에 없습니다."
            )
        sensors = self.config.sensor_columns or _infer_sensor_columns(
            X, self.config.timestamp_column
        )
        missing = [column for column in sensors if column not in X.columns]
        if missing:
            raise ValueError(f"센서 열이 입력 데이터에 없습니다: {missing}")
        if not sensors:
            raise ValueError("분석할 숫자형 센서 열을 찾지 못했습니다.")
        self.sensor_columns_ = list(sensors)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "sensor_columns_"):
            raise RuntimeError("transform 전에 fit을 호출해야 합니다.")

        timestamp_column = self.config.timestamp_column
        frame = X[[timestamp_column, *self.sensor_columns_]].copy()
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
        frame = frame.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
        if frame.empty:
            raise ValueError("유효한 시간 값이 없습니다.")

        for sensor in self.sensor_columns_:
            frame[sensor] = pd.to_numeric(frame[sensor], errors="coerce")

        # 동일 시각의 중복 샘플은 원시 단위 평균으로 한 지점에 집계한다.
        frame = frame.groupby(timestamp_column, as_index=False, sort=True)[
            self.sensor_columns_
        ].mean()
        if self.config.resample_rule:
            frame = (
                frame.set_index(timestamp_column)
                .resample(self.config.resample_rule)
                .mean()
                .reset_index()
            )
        if len(frame) < 8:
            raise ValueError("RUL 분석에는 최소 8개 시계열 지점이 필요합니다.")

        elapsed_days = (
            frame[timestamp_column] - frame[timestamp_column].iloc[0]
        ).dt.total_seconds() / 86400.0
        elapsed_delta = elapsed_days.diff().replace(0, np.nan)

        for sensor in self.sensor_columns_:
            raw = frame[sensor]
            clean = raw.interpolate(
                method="linear",
                limit=self.config.interpolation_limit or None,
                limit_direction="both",
            )
            clean = clean.ffill().bfill()
            if clean.isna().all():
                raise ValueError(f"센서 '{sensor}'에 유효한 측정값이 없습니다.")

            median = clean.rolling(
                self.config.median_window, center=True, min_periods=1
            ).median()
            smooth = median.ewm(span=self.config.ewma_span, adjust=False).mean()
            rolling = smooth.rolling(self.config.rolling_window, min_periods=2)

            frame[f"{sensor}__clean"] = clean
            frame[f"{sensor}__smooth"] = smooth
            frame[f"{sensor}__rolling_mean"] = rolling.mean().fillna(smooth)
            frame[f"{sensor}__rolling_std"] = rolling.std(ddof=1).fillna(0.0)
            frame[f"{sensor}__gradient"] = (
                smooth.diff().div(elapsed_delta).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
            frame[f"{sensor}__cum_degradation"] = smooth.diff().abs().fillna(0.0).cumsum()

        return frame.reset_index(drop=True)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        del input_features
        if not hasattr(self, "sensor_columns_"):
            raise RuntimeError("get_feature_names_out 전에 fit을 호출해야 합니다.")
        names = [self.config.timestamp_column, *self.sensor_columns_]
        for sensor in self.sensor_columns_:
            names.extend(f"{sensor}{suffix}" for suffix in DERIVED_SUFFIXES[:-1])
        return np.asarray(names, dtype=object)


@dataclass(slots=True)
class SensorControlLimits:
    sensor: str
    baseline_mean: float
    baseline_std: float
    direction: int
    warning_threshold: float
    failure_threshold: float
    onset_index: int | None
    warning_index: int | None
    onset_time: pd.Timestamp | None
    latest_zscore: float
    first_failure_index: int | None

    @property
    def direction_label(self) -> str:
        return "상승" if self.direction > 0 else "하강"


@dataclass(slots=True)
class OnsetDetectionResult:
    status: str
    baseline_end_index: int
    onset_index: int | None
    onset_time: pd.Timestamp | None
    active_sensors: list[str]
    sensor_limits: dict[str, SensorControlLimits]
    health_score: pd.Series
    catastrophic_sensor: str | None = None


def _first_sustained(mask: np.ndarray, length: int, start: int = 0) -> int | None:
    if length <= 1:
        matches = np.flatnonzero(mask[start:])
        return int(start + matches[0]) if len(matches) else None
    values = np.asarray(mask, dtype=np.int8)
    run = np.convolve(values, np.ones(length, dtype=np.int8), mode="valid")
    matches = np.flatnonzero((run >= length) & (np.arange(len(run)) >= start))
    return int(matches[0]) if len(matches) else None


def _estimate_inflection(
    values: np.ndarray,
    baseline_mean: float,
    baseline_start: int,
    warning_index: int,
    min_active_points: int,
) -> int:
    """확인된 2σ 경고점에서 역추적해 기준→열화 변곡점을 찾는다.

    후보점 이전은 건강 기준 평균, 이후는 1차 열화 추세로 설명하고 전체
    제곱오차가 가장 작은 분할을 선택한다. 충분한 후속 구간이 없으면
    통계 경고점 자체를 반환한다.
    """

    search_end = warning_index - max(4, min_active_points // 2)
    if search_end <= baseline_start:
        return warning_index
    end = min(len(values), warning_index + max(3, min_active_points // 2))
    best_index = warning_index
    best_score = float("inf")
    for candidate in range(baseline_start, search_end + 1):
        before = values[baseline_start:candidate]
        after = values[candidate:end]
        if len(before) < 2 or len(after) < 4:
            continue
        x_after = np.arange(len(after), dtype=float)
        coefficients = np.polyfit(x_after, after, deg=1)
        fitted_after = np.polyval(coefficients, x_after)
        sse_before = float(np.sum((before - baseline_mean) ** 2))
        sse_after = float(np.sum((after - fitted_after) ** 2))
        score = (sse_before + sse_after) / (len(before) + len(after))
        if score < best_score:
            best_score = score
            best_index = candidate
    return best_index


class OnsetDetector:
    """건강 기준 분포로부터 2σ 경고와 3σ 고장 경계를 탐지한다."""

    def __init__(self, config: OnsetConfig | None = None):
        self.config = config or OnsetConfig()

    def detect(
        self,
        frame: pd.DataFrame,
        sensor_columns: list[str],
        timestamp_column: str = "timestamp",
    ) -> OnsetDetectionResult:
        self.config.validate()
        n_rows = len(frame)
        if n_rows < 8:
            raise ValueError("열화 시작점 탐지에는 최소 8개 지점이 필요합니다.")

        available_for_baseline = max(5, n_rows - self.config.consecutive_points - 1)
        baseline_count = max(
            self.config.min_baseline_points,
            int(np.ceil(n_rows * self.config.baseline_fraction)),
        )
        baseline_count = min(baseline_count, available_for_baseline)

        limits: dict[str, SensorControlLimits] = {}
        zscores: list[np.ndarray] = []
        active_sensors: list[str] = []
        onset_candidates: list[int] = []
        catastrophic_sensor: str | None = None

        for sensor in sensor_columns:
            smooth_column = f"{sensor}__smooth"
            if smooth_column not in frame.columns:
                raise ValueError(f"평활화 열이 없습니다: {smooth_column}")
            values = frame[smooth_column].to_numpy(dtype=float)
            # 관리 한계는 원시 계측 분산을 보존한 건강 구간에서 계산한다.
            # 평활 신호는 이상 판정에만 사용해 노이즈 오경보를 줄인다.
            baseline_source = frame[f"{sensor}__clean"].to_numpy(dtype=float)
            baseline = baseline_source[:baseline_count]
            mean = float(np.nanmean(baseline))
            measured_std = float(np.nanstd(baseline, ddof=1))
            scale_floor = max(abs(mean) * self.config.scale_floor_ratio, 1e-9)
            std = max(measured_std, scale_floor)
            z = (values - mean) / std
            zscores.append(np.abs(z))

            warning_index = _first_sustained(
                np.abs(z) >= self.config.warning_sigma,
                self.config.consecutive_points,
                start=baseline_count,
            )
            if warning_index is not None:
                onset = _estimate_inflection(
                    values,
                    mean,
                    baseline_count,
                    warning_index,
                    self.config.min_active_points,
                )
                active_sensors.append(sensor)
                onset_candidates.append(onset)
                tail = values[warning_index:]
                delta = float(np.nanmedian(tail[-min(len(tail), 5) :]) - mean)
                if abs(delta) <= scale_floor:
                    delta = float(values[-1] - values[onset])
                direction = 1 if delta >= 0 else -1
            else:
                onset = None
                direction = 1 if values[-1] >= mean else -1

            failure_threshold = self.config.failure_threshold_overrides.get(
                sensor,
                mean + direction * self.config.failure_sigma * std,
            )
            failure = _first_sustained(
                direction * values >= direction * failure_threshold,
                self.config.consecutive_points,
                start=baseline_count,
            )
            raw_z = (baseline_source - mean) / std
            raw_jumps = np.abs(np.diff(raw_z, prepend=raw_z[0]))
            if (
                failure is not None
                and np.nanmax(raw_jumps[baseline_count : failure + 1])
                >= self.config.catastrophic_jump_sigma
                and catastrophic_sensor is None
            ):
                catastrophic_sensor = sensor

            limits[sensor] = SensorControlLimits(
                sensor=sensor,
                baseline_mean=mean,
                baseline_std=std,
                direction=direction,
                warning_threshold=mean + direction * self.config.warning_sigma * std,
                failure_threshold=float(failure_threshold),
                onset_index=onset,
                warning_index=warning_index,
                onset_time=(
                    pd.Timestamp(frame[timestamp_column].iloc[onset])
                    if onset is not None
                    else None
                ),
                latest_zscore=float(z[-1]),
                first_failure_index=failure,
            )
            frame[f"{sensor}__zscore"] = z

        health_score = pd.Series(
            np.nanmax(np.vstack(zscores), axis=0),
            index=frame.index,
            name="통합_건강_이상점수",
        )
        global_onset = min(onset_candidates) if onset_candidates else None
        onset_time = (
            pd.Timestamp(frame[timestamp_column].iloc[global_onset])
            if global_onset is not None
            else None
        )

        if catastrophic_sensor:
            status = "급격 고장"
        elif any(limits[s].first_failure_index is not None for s in active_sensors):
            status = "고장 한계 도달"
        elif active_sensors:
            status = "활성 열화"
        else:
            status = "정상/드리프트 없음"

        return OnsetDetectionResult(
            status=status,
            baseline_end_index=baseline_count - 1,
            onset_index=global_onset,
            onset_time=onset_time,
            active_sensors=active_sensors,
            sensor_limits=limits,
            health_score=health_score,
            catastrophic_sensor=catastrophic_sensor,
        )
