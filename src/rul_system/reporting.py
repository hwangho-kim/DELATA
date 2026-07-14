"""원시 단위 기반 한글 RUL 진단 그림, JSON, CSV, HTML 보고서."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ReportingConfig
from .pipeline import EquipmentRULResult


class RULReportGenerator:
    """RUL 결과를 엔지니어가 해석 가능한 원시 센서 단위로 출력한다."""

    def __init__(self, config: ReportingConfig | None = None):
        self.config = config or ReportingConfig()
        self._configure_font()

    def _configure_font(self) -> None:
        installed = {font.name for font in fm.fontManager.ttflist}
        selected = next(
            (font for font in self.config.korean_font_candidates if font in installed),
            "DejaVu Sans",
        )
        plt.rcParams["font.family"] = selected
        plt.rcParams["axes.unicode_minus"] = False

    def generate(self, result: EquipmentRULResult, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "rul_summary.json"
        plot_path = output / "rul_diagnostic.png"
        html_path = output / "rul_report.html"

        summary = result.to_summary_dict()
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._plot(result, plot_path)
        csv_paths = self._write_forecasts(result, output)
        self._write_html(result, html_path, plot_path.name, csv_paths)
        return {
            "요약_JSON": summary_path,
            "진단_그래프": plot_path,
            "한글_HTML_보고서": html_path,
            **{f"예측_CSV_{sensor}": path for sensor, path in csv_paths.items()},
        }

    def _plot(self, result: EquipmentRULResult, path: Path) -> None:
        sensor_count = len(result.sensor_columns)
        fig, axes = plt.subplots(
            sensor_count,
            1,
            figsize=(
                self.config.figure_width,
                self.config.subplot_height * sensor_count,
            ),
            squeeze=False,
        )
        times = pd.to_datetime(result.processed_data[result.timestamp_column])
        analysis_time = result.analysis_time

        for axis, sensor in zip(axes[:, 0], result.sensor_columns, strict=True):
            raw = result.processed_data[sensor]
            smooth = result.processed_data[f"{sensor}__smooth"]
            limit = result.detection.sensor_limits[sensor]
            sensor_result = result.sensor_results[sensor]

            axis.plot(times, raw, color="#8a94a6", linewidth=1.0, alpha=0.65, label="원시 FDC 데이터")
            axis.plot(times, smooth, color="#1665d8", linewidth=2.0, label="EWMA 평활 추세")
            axis.axhline(
                limit.warning_threshold,
                color="#e69f00",
                linestyle="--",
                linewidth=1.4,
                label="2σ 경고 한계",
            )
            axis.axhline(
                limit.failure_threshold,
                color="#d62728",
                linestyle="--",
                linewidth=1.6,
                label="3σ 고장 한계",
            )
            if limit.onset_time is not None:
                axis.axvline(
                    limit.onset_time,
                    color="#7b2cbf",
                    linestyle=":",
                    linewidth=1.8,
                    label="탐지된 열화 시작점",
                )
            if sensor_result.fit_times is not None and sensor_result.fitted_raw is not None:
                axis.plot(
                    sensor_result.fit_times,
                    sensor_result.fitted_raw,
                    color="#009e73",
                    linewidth=2.1,
                    label=f"적합 곡선 ({sensor_result.model_name})",
                )
            if sensor_result.future_times is not None and sensor_result.future_raw is not None:
                future = sensor_result.future_times
                prediction = sensor_result.future_raw
                if sensor_result.failure_time is None:
                    observed_days = max(float((analysis_time - times.iloc[0]).total_seconds() / 86400), 1.0)
                    display_end = analysis_time + pd.to_timedelta(max(30.0, observed_days), unit="D")
                    display_mask = np.asarray(future <= display_end)
                else:
                    display_mask = np.ones(len(future), dtype=bool)
                finite = np.isfinite(prediction) & display_mask
                axis.plot(
                    future[finite],
                    prediction[finite],
                    color="#cc79a7",
                    linewidth=2.0,
                    linestyle="-.",
                    label="외삽 예측 (원시 단위)",
                )
                displayed_future = future[finite]
                if len(displayed_future):
                    axis.axvspan(
                        analysis_time,
                        displayed_future[-1],
                        color="#cc79a7",
                        alpha=0.06,
                        label="외삽 구간",
                    )
            if sensor_result.failure_time is not None:
                axis.scatter(
                    [sensor_result.failure_time],
                    [limit.failure_threshold],
                    s=58,
                    marker="X",
                    color="#d62728",
                    zorder=5,
                    label="예상/관측 고장점",
                )

            formula = sensor_result.formula or f"모델식 없음 — {sensor_result.status}"
            axis.text(
                0.015,
                0.96,
                f"모델식: {formula}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
            )
            axis.set_title(f"센서: {sensor}  |  상태: {sensor_result.status}", loc="left")
            axis.set_ylabel(f"{sensor} (원시 단위)")
            axis.grid(True, alpha=0.22)
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axis.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=8.5,
            )

        axes[-1, 0].set_xlabel("시간")
        failure_text = result.estimated_failure_date or "산출 불가"
        rul_text = f"{result.rul_days:.3f}일" if result.rul_days is not None else "산출 불가"
        fig.suptitle(
            f"FDC 자동 RUL 진단 | 예상 고장일: {failure_text} | 잔여수명: {rul_text}",
            fontsize=15,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.975))
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _write_forecasts(result: EquipmentRULResult, output: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for sensor, sensor_result in result.sensor_results.items():
            if sensor_result.future_times is None or sensor_result.future_raw is None:
                continue
            safe_sensor = "".join(char if char.isalnum() or char in "-_" else "_" for char in sensor)
            path = output / f"forecast_{safe_sensor}.csv"
            limit = result.detection.sensor_limits[sensor]
            pd.DataFrame(
                {
                    "예측_시각": sensor_result.future_times,
                    f"{sensor}_예측값_원시단위": sensor_result.future_raw,
                    "2시그마_경고한계_원시단위": limit.warning_threshold,
                    "3시그마_고장한계_원시단위": limit.failure_threshold,
                }
            ).to_csv(path, index=False, encoding="utf-8-sig")
            paths[sensor] = path
        return paths

    @staticmethod
    def _write_html(
        result: EquipmentRULResult,
        path: Path,
        plot_name: str,
        csv_paths: dict[str, Path],
    ) -> None:
        rows: list[str] = []
        for sensor, sensor_result in result.sensor_results.items():
            limit = result.detection.sensor_limits[sensor]
            csv_link = (
                f'<a href="{escape(csv_paths[sensor].name)}">예측 CSV</a>'
                if sensor in csv_paths
                else "-"
            )
            rul_cell = (
                f"{sensor_result.rul_days:.3f}"
                if sensor_result.rul_days is not None
                else "-"
            )
            rows.append(
                "<tr>"
                f"<td>{escape(sensor)}</td>"
                f"<td>{escape(sensor_result.status)}</td>"
                f"<td>{escape(sensor_result.model_name or '-')}</td>"
                f"<td>{limit.warning_threshold:.6g}</td>"
                f"<td>{limit.failure_threshold:.6g}</td>"
                f"<td>{rul_cell}</td>"
                f"<td>{csv_link}</td>"
                "</tr>"
            )

        warning_items = "".join(f"<li>{escape(message)}</li>" for message in result.warnings)
        if not warning_items:
            warning_items = "<li>별도 주의사항 없음</li>"
        failure_date = result.estimated_failure_date or "산출 불가"
        rul_days = f"{result.rul_days:.3f}일" if result.rul_days is not None else "산출 불가"
        critical = escape(result.critical_sensor or "없음")
        html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FDC 자동 RUL 진단 보고서</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif; margin: 0; background:#f4f6f9; color:#182230; }}
    main {{ max-width: 1180px; margin: 32px auto; background:white; padding:32px; border-radius:14px; box-shadow:0 8px 30px #1b263b18; }}
    h1 {{ margin-top:0; }} .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
    .card {{ background:#eef4ff; padding:16px; border-radius:10px; }} .label {{ color:#5d6b82; font-size:13px; }}
    .value {{ font-size:20px; font-weight:700; margin-top:6px; }} img {{ width:100%; margin:24px 0; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:10px; border-bottom:1px solid #dde3ec; text-align:left; }}
    th {{ background:#f6f8fb; }} code {{ word-break:break-all; }}
  </style>
</head>
<body><main>
  <h1>FDC 자동 잔여수명(RUL) 진단 보고서</h1>
  <div class="cards">
    <div class="card"><div class="label">시스템 상태</div><div class="value">{escape(result.status)}</div></div>
    <div class="card"><div class="label">예상 고장일</div><div class="value">{failure_date}</div></div>
    <div class="card"><div class="label">잔여수명</div><div class="value">{rul_days}</div></div>
    <div class="card"><div class="label">주요 위험 센서</div><div class="value">{critical}</div></div>
  </div>
  <img src="{escape(plot_name)}" alt="RUL 진단 그래프">
  <h2>센서별 결과</h2>
  <table><thead><tr><th>센서</th><th>상태</th><th>선택 모델</th><th>2σ 경고</th><th>3σ 고장</th><th>RUL(일)</th><th>자료</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
  <h2>해석 시 주의사항</h2><ul>{warning_items}</ul>
  <p>모든 센서 값·임계값·적합 및 외삽 궤적은 원시 계측 단위로 표시되었습니다.</p>
</main></body></html>"""
        path.write_text(html, encoding="utf-8")
