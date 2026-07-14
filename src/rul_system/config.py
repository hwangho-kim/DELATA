"""RUL 시스템의 중앙 설정 객체."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PreprocessingConfig:
    """입력 정리, 평활화, 특징 생성 설정."""

    timestamp_column: str = "timestamp"
    sensor_columns: list[str] | None = None
    ewma_span: int = 8
    median_window: int = 3
    rolling_window: int = 10
    interpolation_limit: int = 5
    resample_rule: str | None = None

    def validate(self) -> None:
        if self.ewma_span < 1:
            raise ValueError("EWMA span은 1 이상이어야 합니다.")
        if self.median_window < 1 or self.rolling_window < 2:
            raise ValueError("필터/롤링 윈도우 설정이 올바르지 않습니다.")
        if self.interpolation_limit < 0:
            raise ValueError("보간 한도는 음수일 수 없습니다.")


@dataclass(slots=True)
class OnsetConfig:
    """건강 기준 구간과 통계적 관리 한계 설정."""

    baseline_fraction: float = 0.25
    min_baseline_points: int = 20
    warning_sigma: float = 2.0
    failure_sigma: float = 3.0
    consecutive_points: int = 3
    min_active_points: int = 8
    catastrophic_jump_sigma: float = 8.0
    scale_floor_ratio: float = 1e-6
    failure_threshold_overrides: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0 < self.baseline_fraction < 1:
            raise ValueError("baseline_fraction은 0과 1 사이여야 합니다.")
        if self.min_baseline_points < 5:
            raise ValueError("건강 기준 구간은 최소 5개 지점이 필요합니다.")
        if not 0 < self.warning_sigma < self.failure_sigma:
            raise ValueError("0 < 경고 sigma < 고장 sigma 조건이 필요합니다.")
        if self.consecutive_points < 1:
            raise ValueError("연속 이상 지점 수는 1 이상이어야 합니다.")
        if any(not math.isfinite(value) for value in self.failure_threshold_overrides.values()):
            raise ValueError("센서별 고장 한계 재정의 값은 유한한 숫자여야 합니다.")


@dataclass(slots=True)
class AutoMLConfig:
    """시계열 교차 검증, 모델 탐색, 외삽 설정."""

    cv_splits: int = 3
    random_state: int = 42
    max_extrapolation_days: float = 3650.0
    extrapolation_grid_points: int = 5000
    trend_penalty_weight: float = 0.75
    enabled_models: list[str] = field(
        default_factory=lambda: [
            "선형 회귀",
            "다항 회귀",
            "지수 성장",
            "SVR",
            "Gradient Boosting",
            "XGBoost",
        ]
    )
    enable_xgboost_if_available: bool = True

    def validate(self) -> None:
        if self.cv_splits < 2:
            raise ValueError("시계열 교차 검증 분할 수는 2 이상이어야 합니다.")
        if self.max_extrapolation_days <= 0:
            raise ValueError("최대 외삽 기간은 양수여야 합니다.")
        if self.extrapolation_grid_points < 100:
            raise ValueError("외삽 격자 수는 100 이상이어야 합니다.")


@dataclass(slots=True)
class ReportingConfig:
    """한글 진단 보고서 출력 설정."""

    dpi: int = 150
    figure_width: float = 14.0
    subplot_height: float = 4.2
    korean_font_candidates: list[str] = field(
        default_factory=lambda: [
            "AppleGothic",
            "Malgun Gothic",
            "NanumGothic",
            "Noto Sans CJK KR",
            "DejaVu Sans",
        ]
    )


@dataclass(slots=True)
class RULSystemConfig:
    """전체 파이프라인을 위한 직렬화 가능한 설정."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def validate(self) -> None:
        self.preprocessing.validate()
        self.onset.validate()
        self.automl.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RULSystemConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            preprocessing=PreprocessingConfig(**payload.get("preprocessing", {})),
            onset=OnsetConfig(**payload.get("onset", {})),
            automl=AutoMLConfig(**payload.get("automl", {})),
            reporting=ReportingConfig(**payload.get("reporting", {})),
        )
        config.validate()
        return config
