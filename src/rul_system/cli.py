"""FDC RUL 시스템 명령행 인터페이스."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import RULSystemConfig
from .pipeline import RULPredictor
from .preprocessing_and_onset import load_fdc_data
from .reporting import RULReportGenerator
from .sample_data import make_synthetic_fdc_data


def _run(data_path: Path, output: Path, config: RULSystemConfig) -> int:
    data = load_fdc_data(data_path)
    result = RULPredictor(config).predict(data)
    artifacts = RULReportGenerator(config.reporting).generate(result, output)
    print("\n=== FDC 자동 RUL 분석 결과 ===")
    print(f"시스템 상태: {result.status}")
    print(f"예상 고장일: {result.estimated_failure_date or '산출 불가'}")
    print(
        f"잔여수명: {result.rul_days:.3f}일 / {result.rul_hours:.2f}시간"
        if result.rul_days is not None and result.rul_hours is not None
        else "잔여수명: 산출 불가"
    )
    print(f"주요 위험 센서: {result.critical_sensor or '없음'}")
    print("\n생성 파일:")
    for label, path in artifacts.items():
        print(f"- {label}: {path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rul-system",
        description="반도체 FDC 데이터용 자동 잔여수명(RUL) 예측 시스템",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="CSV/Parquet FDC 데이터를 분석합니다.")
    analyze.add_argument("input", type=Path, help="입력 CSV 또는 Parquet 파일")
    analyze.add_argument("--output", type=Path, default=Path("rul_output"), help="보고서 출력 폴더")
    analyze.add_argument("--config", type=Path, help="JSON 설정 파일")
    analyze.add_argument("--timestamp", help="시간 열 이름")
    analyze.add_argument("--sensors", nargs="+", help="분석할 센서 열 목록")

    demo = subparsers.add_parser("demo", help="합성 FDC 데이터로 전체 흐름을 실행합니다.")
    demo.add_argument("--output", type=Path, default=Path("demo_output"), help="출력 폴더")
    demo.add_argument("--sudden-failure", action="store_true", help="급격 고장 예제를 생성합니다.")
    demo.add_argument("--no-drift", action="store_true", help="무드리프트 예제를 생성합니다.")

    init = subparsers.add_parser("init-config", help="기본 한글 설정 JSON을 생성합니다.")
    init.add_argument("--path", type=Path, default=Path("rul_config.json"), help="설정 파일 경로")

    web = subparsers.add_parser("web", help="현업 사용자용 RUL 웹 화면을 실행합니다.")
    web.add_argument("--host", default="127.0.0.1", help="바인딩 호스트")
    web.add_argument("--port", type=int, default=8000, help="바인딩 포트")
    web.add_argument("--reload", action="store_true", help="개발 중 자동 재시작")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-config":
        RULSystemConfig().save(args.path)
        print(f"기본 설정 파일을 생성했습니다: {args.path.resolve()}")
        return 0

    if args.command == "web":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit(
                "웹 의존성이 없습니다. 'pip install -e .[web]'로 설치하세요."
            ) from exc
        print(f"FDC 자동 RUL 웹 화면: http://{args.host}:{args.port}")
        uvicorn.run(
            "rul_system.webapp:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    if args.command == "demo":
        args.output.mkdir(parents=True, exist_ok=True)
        sample_path = args.output / "synthetic_fdc.csv"
        make_synthetic_fdc_data(
            sudden_failure=args.sudden_failure,
            no_drift=args.no_drift,
        ).to_csv(sample_path, index=False, encoding="utf-8-sig")
        return _run(sample_path, args.output, RULSystemConfig())

    config = RULSystemConfig.load(args.config) if args.config else RULSystemConfig()
    if args.timestamp:
        config.preprocessing.timestamp_column = args.timestamp
    if args.sensors:
        config.preprocessing.sensor_columns = args.sensors
    config.validate()
    return _run(args.input, args.output, config)


if __name__ == "__main__":
    raise SystemExit(main())
