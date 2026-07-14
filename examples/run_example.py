"""python examples/run_example.py 로 실행하는 최소 예제."""

from pathlib import Path

from rul_system import RULPredictor, RULReportGenerator
from rul_system.sample_data import make_synthetic_fdc_data


data = make_synthetic_fdc_data()
result = RULPredictor().predict(data)
files = RULReportGenerator().generate(result, Path("example_output"))

print(f"예상 고장일: {result.estimated_failure_date or '산출 불가'}")
print(f"잔여수명(일): {result.rul_days if result.rul_days is not None else '산출 불가'}")
for name, path in files.items():
    print(f"{name}: {path}")
