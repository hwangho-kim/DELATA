"""현업 사용자용 FastAPI RUL 분석 웹 애플리케이션."""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from starlette.concurrency import run_in_threadpool
except ImportError as exc:  # pragma: no cover - 설치 안내 경로
    raise ImportError(
        "웹 화면 의존성이 없습니다. 'pip install -e .[web]'로 설치하세요."
    ) from exc

from .config import AutoMLConfig, RULSystemConfig
from .pipeline import EquipmentRULResult, RULPredictor
from .sample_data import make_synthetic_fdc_data


STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _read_uploaded_data(content: bytes, filename: str) -> pd.DataFrame:
    if not content:
        raise ValueError("업로드된 파일이 비어 있습니다.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("파일 크기는 30MB 이하여야 합니다.")
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(content)
    if suffix in {".csv", ".txt", ""}:
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp949"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError("CSV 문자 인코딩을 해석하지 못했습니다.") from last_error
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(buffer)
        except ImportError as exc:
            raise ValueError(
                "Parquet 엔진이 없습니다. 'pip install -e .[parquet]'가 필요합니다."
            ) from exc
    raise ValueError("웹 업로드는 CSV와 Parquet 파일만 지원합니다.")


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def _number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    number = float(value)
    return number if np.isfinite(number) else None


def _timestamps(values: Any) -> list[str]:
    return [pd.Timestamp(value).isoformat() for value in values]


def _tone(status: str) -> str:
    if "고장" in status and "예측 완료" not in status:
        return "danger"
    if "활성 열화" in status or "불확정" in status:
        return "warning"
    if "예측 완료" in status:
        return "forecast"
    return "normal"


def _build_process_steps(result: EquipmentRULResult, duration_seconds: float) -> list[dict[str, str]]:
    baseline_end = pd.Timestamp(
        result.processed_data[result.timestamp_column].iloc[result.detection.baseline_end_index]
    )
    selected_models = [
        f"{sensor}: {sensor_result.model_name}"
        for sensor, sensor_result in result.sensor_results.items()
        if sensor_result.model_name
    ]
    active = ", ".join(result.detection.active_sensors) or "없음"
    onset = (
        result.detection.onset_time.strftime("%Y-%m-%d %H:%M")
        if result.detection.onset_time is not None
        else "탐지되지 않음"
    )
    prediction = (
        f"{result.estimated_failure_date}, {result.rul_days:.3f}일"
        if result.estimated_failure_date is not None and result.rul_days is not None
        else "현재 근거로 산출 불가"
    )
    return [
        {
            "단계": "01",
            "이름": "입력 검증",
            "상태": "완료",
            "설명": f"{len(result.processed_data):,}개 시점 · {len(result.sensor_columns)}개 센서",
        },
        {
            "단계": "02",
            "이름": "노이즈 저감·특징 생성",
            "상태": "완료",
            "설명": "Median 필터, EWMA, 롤링 통계, 기울기, 누적 변화량",
        },
        {
            "단계": "03",
            "이름": "건강 기준·관리 한계",
            "상태": "완료",
            "설명": f"{result.detection.baseline_end_index + 1:,}개 기준점 · 종료 {baseline_end:%Y-%m-%d %H:%M}",
        },
        {
            "단계": "04",
            "이름": "열화 시작점 탐지",
            "상태": "주의" if result.detection.active_sensors else "완료",
            "설명": f"열화 센서 {active} · 변곡점 {onset}",
        },
        {
            "단계": "05",
            "이름": "AutoML 궤적 선택",
            "상태": "완료" if selected_models else "해당 없음",
            "설명": " / ".join(selected_models) if selected_models else "모델링할 열화 궤적 없음",
        },
        {
            "단계": "06",
            "이름": "고장 교차·RUL 계산",
            "상태": "완료" if result.estimated_failure_time is not None else "주의",
            "설명": f"{prediction} · 총 분석 {duration_seconds:.2f}초",
        },
    ]


def _sensor_payload(result: EquipmentRULResult, sensor: str) -> dict[str, Any]:
    frame = result.processed_data
    sensor_result = result.sensor_results[sensor]
    limit = result.detection.sensor_limits[sensor]
    observed_index = _sample_indices(len(frame), 1200)
    observed_times = pd.to_datetime(frame[result.timestamp_column]).iloc[observed_index]
    warning_time = (
        pd.Timestamp(frame[result.timestamp_column].iloc[limit.warning_index]).isoformat()
        if limit.warning_index is not None
        else None
    )

    fit = {"시간": [], "값": []}
    if sensor_result.fit_times is not None and sensor_result.fitted_raw is not None:
        fit_index = _sample_indices(len(sensor_result.fit_times), 800)
        fit = {
            "시간": _timestamps(sensor_result.fit_times[fit_index]),
            "값": [_number(value) for value in sensor_result.fitted_raw[fit_index]],
        }
    future = {"시간": [], "값": []}
    if sensor_result.future_times is not None and sensor_result.future_raw is not None:
        future_index = _sample_indices(len(sensor_result.future_times), 800)
        future = {
            "시간": _timestamps(sensor_result.future_times[future_index]),
            "값": [_number(value) for value in sensor_result.future_raw[future_index]],
        }

    leaderboard = [row.to_dict() for row in sensor_result.leaderboard[:15]]
    latest = frame.iloc[-1]
    return {
        "센서": sensor,
        "상태": sensor_result.status,
        "상태_톤": _tone(sensor_result.status),
        "열화_방향": limit.direction_label,
        "열화_시작시각": limit.onset_time.isoformat() if limit.onset_time is not None else None,
        "2시그마_최초확인시각": warning_time,
        "예상_고장시각": (
            sensor_result.failure_time.isoformat()
            if sensor_result.failure_time is not None
            else None
        ),
        "잔여수명_일": sensor_result.rul_days,
        "잔여수명_시간": sensor_result.rul_hours,
        "기준_평균": limit.baseline_mean,
        "기준_표준편차": limit.baseline_std,
        "경고_한계": limit.warning_threshold,
        "고장_한계": limit.failure_threshold,
        "최신_zscore": limit.latest_zscore,
        "모델": sensor_result.model_name,
        "모델식": sensor_result.formula,
        "모델_파라미터": sensor_result.model_parameters,
        "평가지표": sensor_result.metrics,
        "AutoML_순위표": leaderboard,
        "최신_특징": {
            "원시값": _number(latest[sensor]),
            "EWMA": _number(latest[f"{sensor}__smooth"]),
            "롤링_평균": _number(latest[f"{sensor}__rolling_mean"]),
            "롤링_표준편차": _number(latest[f"{sensor}__rolling_std"]),
            "일당_기울기": _number(latest[f"{sensor}__gradient"]),
            "누적_변화량": _number(latest[f"{sensor}__cum_degradation"]),
        },
        "관측": {
            "시간": _timestamps(observed_times),
            "원시값": [_number(value) for value in frame[sensor].iloc[observed_index]],
            "평활값": [
                _number(value) for value in frame[f"{sensor}__smooth"].iloc[observed_index]
            ],
        },
        "적합": fit,
        "외삽": future,
    }


def _build_dashboard_payload(
    result: EquipmentRULResult,
    source: pd.DataFrame,
    filename: str,
    file_size: int,
    duration_seconds: float,
) -> dict[str, Any]:
    frame = result.processed_data
    timestamp = result.timestamp_column
    parsed_time = pd.to_datetime(source[timestamp], errors="coerce")
    duplicate_count = int(parsed_time.duplicated().sum())
    invalid_time_count = int(parsed_time.isna().sum())
    missing = {
        sensor: int(pd.to_numeric(source[sensor], errors="coerce").isna().sum())
        for sensor in result.sensor_columns
    }
    time_values = pd.to_datetime(frame[timestamp])
    interval_hours = (
        float(time_values.diff().dt.total_seconds().median() / 3600.0)
        if len(time_values) > 1
        else None
    )
    health_index = _sample_indices(len(result.detection.health_score), 1200)
    preview_frame = frame[[timestamp, *result.sensor_columns]].tail(40).copy()
    preview_frame[timestamp] = pd.to_datetime(preview_frame[timestamp]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    preview = [
        {key: _number(value) if key != timestamp else value for key, value in row.items()}
        for row in preview_frame.to_dict(orient="records")
    ]

    return {
        "요약": result.to_summary_dict(),
        "화면": {
            "상태": result.status,
            "상태_톤": _tone(result.status),
            "예상_고장일": result.estimated_failure_date,
            "예상_고장시각": (
                result.estimated_failure_time.isoformat()
                if result.estimated_failure_time is not None
                else None
            ),
            "RUL_일": result.rul_days,
            "RUL_시간": result.rul_hours,
            "주요_위험센서": result.critical_sensor,
            "분석_기준시각": result.analysis_time.isoformat(),
            "분석_소요초": round(duration_seconds, 3),
        },
        "데이터_품질": {
            "파일명": filename,
            "파일_크기_MB": round(file_size / (1024 * 1024), 3),
            "입력_행수": len(source),
            "처리_행수": len(frame),
            "센서_수": len(result.sensor_columns),
            "센서_목록": result.sensor_columns,
            "시작시각": time_values.iloc[0].isoformat(),
            "종료시각": time_values.iloc[-1].isoformat(),
            "중앙_수집주기_시간": interval_hours,
            "중복_시각수": duplicate_count,
            "무효_시각수": invalid_time_count,
            "센서별_결측수": missing,
        },
        "분석_단계": _build_process_steps(result, duration_seconds),
        "센서": {
            sensor: _sensor_payload(result, sensor) for sensor in result.sensor_columns
        },
        "건강_이상점수": {
            "시간": _timestamps(time_values.iloc[health_index]),
            "점수": [
                _number(value) for value in result.detection.health_score.iloc[health_index]
            ],
            "경고_기준": 2.0,
            "고장_기준": 3.0,
            "설명": "센서별 건강 기준 대비 절대 z-score의 최댓값(내부 탐지용 파생 지표)",
        },
        "원시_미리보기": {"열": [timestamp, *result.sensor_columns], "행": preview},
        "주의사항": result.warnings,
    }


def create_app() -> Any:
    app = FastAPI(
        title="FDC 자동 RUL 워크벤치",
        description="반도체 설비 FDC 열화 진단과 잔여수명 예측 웹 화면",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"상태": "정상", "서비스": "FDC 자동 RUL 워크벤치", "버전": "0.2.0"}

    @app.get("/api/sample")
    async def sample(
        mode: str = Query("degradation", pattern="^(degradation|no_drift|sudden_failure)$")
    ) -> Any:
        data = make_synthetic_fdc_data(
            no_drift=mode == "no_drift",
            sudden_failure=mode == "sudden_failure",
        )
        return Response(
            content=data.to_csv(index=False),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="synthetic_fdc.csv"'},
        )

    @app.post("/api/analyze")
    async def analyze(
        request: Request,
        filename: str = Query("fdc_trace.csv", max_length=255),
        timestamp_column: str = Query("timestamp", min_length=1, max_length=128),
        sensor_columns: str = Query("", max_length=2000),
        ewma_span: int = Query(8, ge=1, le=200),
        rolling_window: int = Query(10, ge=2, le=500),
        baseline_fraction: float = Query(0.25, gt=0.05, lt=0.8),
        warning_sigma: float = Query(2.0, gt=0.0, le=10.0),
        failure_sigma: float = Query(3.0, gt=0.0, le=20.0),
        consecutive_points: int = Query(3, ge=1, le=50),
        max_extrapolation_days: float = Query(3650.0, gt=0.0, le=36500.0),
        enabled_models: str = Query(
            "선형 회귀,다항 회귀,지수 성장,SVR,Gradient Boosting",
            max_length=1000,
        ),
    ) -> Any:
        try:
            body = await request.body()
            data = _read_uploaded_data(body, filename)
            sensors = [value.strip() for value in sensor_columns.split(",") if value.strip()]
            models = [value.strip() for value in enabled_models.split(",") if value.strip()]
            config = RULSystemConfig()
            config.preprocessing.timestamp_column = timestamp_column
            config.preprocessing.sensor_columns = sensors or None
            config.preprocessing.ewma_span = ewma_span
            config.preprocessing.rolling_window = rolling_window
            config.onset.baseline_fraction = baseline_fraction
            config.onset.warning_sigma = warning_sigma
            config.onset.failure_sigma = failure_sigma
            config.onset.consecutive_points = consecutive_points
            config.automl = AutoMLConfig(
                cv_splits=3,
                max_extrapolation_days=max_extrapolation_days,
                extrapolation_grid_points=3000,
                enabled_models=models,
            )
            config.validate()
            if warning_sigma >= failure_sigma:
                raise ValueError("경고 sigma는 고장 sigma보다 작아야 합니다.")
            started = time.perf_counter()
            result = await run_in_threadpool(RULPredictor(config).predict, data)
            elapsed = time.perf_counter() - started
            payload = _build_dashboard_payload(result, data, filename, len(body), elapsed)
            return JSONResponse(payload)
        except (ValueError, RuntimeError, KeyError, ImportError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - 운영 안전망
            raise HTTPException(
                status_code=500,
                detail="분석 중 예상하지 못한 오류가 발생했습니다. 입력 열과 데이터 형식을 확인하세요.",
            ) from exc

    return app


app = create_app()
