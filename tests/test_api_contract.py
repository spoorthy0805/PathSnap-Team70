"""Fast, data-independent regression tests for the PathSnap HTTP contract."""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_exposes_model_and_dataset_state():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["n_trips"], int)
    assert "classifier" in payload and "decoder" in payload


def test_trip_catalog_is_a_list():
    response = client.get("/trips")
    assert response.status_code == 200
    assert isinstance(response.json()["trips"], list)


def test_unknown_trip_is_not_found():
    response = client.post("/predict", json={"trip_id": "does-not-exist"})
    assert response.status_code == 404


def test_malformed_trip_ids_are_rejected():
    for value in ("", "../secret", "trip/one", "trip one"):
        response = client.post("/predict", json={"trip_id": value})
        assert response.status_code in (404, 422)


def test_extra_request_fields_are_rejected():
    response = client.post("/predict", json={"trip_id": "unknown", "reference_route": "private"})
    assert response.status_code == 422
