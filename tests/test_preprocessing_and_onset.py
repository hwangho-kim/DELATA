import numpy as np
import pandas as pd

from rul_system.config import OnsetConfig, PreprocessingConfig
from rul_system.preprocessing_and_onset import FDCPreprocessor, OnsetDetector


def test_preprocessing_preserves_raw_values_and_builds_features() -> None:
    timestamps = pd.date_range("2026-01-01", periods=40, freq="h")
    raw = np.linspace(10.0, 11.0, 40)
    raw[8] = np.nan
    data = pd.DataFrame({"timestamp": timestamps, "sensor_a": raw})
    preprocessor = FDCPreprocessor(
        PreprocessingConfig(sensor_columns=["sensor_a"], ewma_span=4, rolling_window=5)
    )

    transformed = preprocessor.fit_transform(data)

    assert np.isnan(transformed.loc[8, "sensor_a"])
    assert np.isfinite(transformed.loc[8, "sensor_a__clean"])
    assert {
        "sensor_a__smooth",
        "sensor_a__rolling_mean",
        "sensor_a__rolling_std",
        "sensor_a__gradient",
        "sensor_a__cum_degradation",
    }.issubset(transformed.columns)


def test_onset_backtracks_from_warning_and_uses_raw_sigma_limits() -> None:
    rng = np.random.default_rng(7)
    n = 120
    onset = 60
    values = 20.0 + rng.normal(0.0, 0.1, n)
    values[onset:] += np.linspace(0.0, 0.35, n - onset)
    data = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=n, freq="h"), "sensor": values}
    )
    processed = FDCPreprocessor(
        PreprocessingConfig(sensor_columns=["sensor"], ewma_span=5)
    ).fit_transform(data)
    result = OnsetDetector(
        OnsetConfig(baseline_fraction=0.25, min_baseline_points=20, consecutive_points=3)
    ).detect(processed, ["sensor"])
    limit = result.sensor_limits["sensor"]

    assert "sensor" in result.active_sensors
    assert limit.warning_index is not None
    assert limit.onset_index is not None
    assert limit.onset_index <= limit.warning_index
    assert np.isclose(
        limit.warning_threshold,
        limit.baseline_mean + limit.direction * 2.0 * limit.baseline_std,
    )
    assert np.isclose(
        limit.failure_threshold,
        limit.baseline_mean + limit.direction * 3.0 * limit.baseline_std,
    )

    overridden = OnsetDetector(
        OnsetConfig(
            baseline_fraction=0.25,
            min_baseline_points=20,
            consecutive_points=3,
            failure_threshold_overrides={"sensor": 20.5},
        )
    ).detect(processed, ["sensor"])
    assert overridden.sensor_limits["sensor"].failure_threshold == 20.5
