from fastapi.testclient import TestClient

from rul_system.sample_data import make_synthetic_fdc_data
from rul_system.webapp import create_app


def test_web_dashboard_and_analysis_api() -> None:
    client = TestClient(create_app())

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["상태"] == "정상"

    index = client.get("/")
    assert index.status_code == 200
    assert "FDC 자동 잔여수명 분석" in index.text

    csv_data = make_synthetic_fdc_data(periods=100).to_csv(index=False).encode()
    response = client.post(
        "/api/analyze",
        params={
            "filename": "sample.csv",
            "enabled_models": "선형 회귀,다항 회귀",
            "max_extrapolation_days": 365,
        },
        content=csv_data,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["화면"]["예상_고장일"] is not None
    assert len(payload["분석_단계"]) == 6
    assert set(payload["센서"]) == {"ChamberPressure", "RFMatch", "CoolantFlow"}
    assert len(payload["원시_미리보기"]["행"]) == 40
    assert payload["센서"]["ChamberPressure"]["관측"]["원시값"]


def test_web_api_returns_korean_validation_error() -> None:
    client = TestClient(create_app())
    csv_data = make_synthetic_fdc_data(periods=60).to_csv(index=False).encode()
    response = client.post(
        "/api/analyze",
        params={"filename": "sample.csv", "timestamp_column": "missing_time"},
        content=csv_data,
    )

    assert response.status_code == 422
    assert "시간 열" in response.json()["detail"]
