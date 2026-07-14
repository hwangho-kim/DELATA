"""설치 검증과 예제 실행을 위한 합성 FDC 데이터."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_fdc_data(
    periods: int = 160,
    frequency: str = "6h",
    seed: int = 42,
    sudden_failure: bool = False,
    no_drift: bool = False,
) -> pd.DataFrame:
    """상승 열화, 하강 열화, 안정 센서를 포함한 재현 가능한 샘플."""

    if periods < 60:
        raise ValueError("합성 데이터는 60개 이상의 지점이 필요합니다.")
    rng = np.random.default_rng(seed)
    time = pd.date_range("2026-01-01", periods=periods, freq=frequency)
    onset = int(periods * 0.45)
    active_length = periods - onset
    progress = np.linspace(0.0, 1.0, active_length)

    chamber_pressure = 100.0 + rng.normal(0.0, 0.22, periods)
    rf_match = 50.0 + rng.normal(0.0, 0.12, periods)
    coolant_flow = 20.0 + rng.normal(0.0, 0.08, periods)
    if not no_drift:
        chamber_pressure[onset:] += 0.50 * progress**1.25
        rf_match[onset:] -= 0.30 * progress**1.10
    if sudden_failure:
        chamber_pressure[-5:] += 3.5

    return pd.DataFrame(
        {
            "timestamp": time,
            "ChamberPressure": chamber_pressure,
            "RFMatch": rf_match,
            "CoolantFlow": coolant_flow,
        }
    )
