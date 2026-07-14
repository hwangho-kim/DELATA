import json

from rul_system.config import AutoMLConfig, RULSystemConfig
from rul_system.pipeline import RULPredictor
from rul_system.reporting import RULReportGenerator
from rul_system.sample_data import make_synthetic_fdc_data


def _fast_config() -> RULSystemConfig:
    config = RULSystemConfig()
    config.automl = AutoMLConfig(
        cv_splits=2,
        max_extrapolation_days=365.0,
        extrapolation_grid_points=1000,
        enabled_models=["선형 회귀", "다항 회귀", "지수 성장"],
    )
    return config


def test_end_to_end_prediction_and_korean_report(tmp_path) -> None:
    result = RULPredictor(_fast_config()).predict(make_synthetic_fdc_data())

    assert result.estimated_failure_time is not None
    assert result.estimated_failure_date is not None
    assert result.rul_days is not None and result.rul_days >= 0
    assert result.critical_sensor in {"ChamberPressure", "RFMatch"}
    assert any(sensor.model_name for sensor in result.sensor_results.values())

    artifacts = RULReportGenerator().generate(result, tmp_path)
    assert artifacts["진단_그래프"].stat().st_size > 10_000
    assert artifacts["한글_HTML_보고서"].exists()
    summary = json.loads(artifacts["요약_JSON"].read_text(encoding="utf-8"))
    assert summary["예상_고장일"] == result.estimated_failure_date
    assert "잔여수명_RUL_일" in summary


def test_no_drift_returns_unknown_rul() -> None:
    result = RULPredictor(_fast_config()).predict(make_synthetic_fdc_data(no_drift=True))

    assert result.status == "정상/드리프트 없음"
    assert result.estimated_failure_time is None
    assert result.rul_days is None


def test_catastrophic_failure_returns_zero_rul() -> None:
    result = RULPredictor(_fast_config()).predict(
        make_synthetic_fdc_data(sudden_failure=True)
    )

    assert result.status == "급격 고장"
    assert result.rul_days == 0.0
    assert result.estimated_failure_time is not None
