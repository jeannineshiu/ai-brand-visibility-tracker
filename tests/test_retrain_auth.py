"""Unit tests for /retrain endpoint authentication."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_key(monkeypatch):
    monkeypatch.setenv("RETRAIN_API_KEY", "test-secret")
    import importlib
    import src.utils.config as cfg
    importlib.reload(cfg)
    with patch("src.recommender.api.RETRAIN_API_KEY", "test-secret"):
        from src.recommender.api import app
        yield TestClient(app)


@pytest.fixture
def client_no_key():
    with patch("src.recommender.api.RETRAIN_API_KEY", ""):
        from src.recommender.api import app
        yield TestClient(app)


class TestRetrainAuth:
    def test_missing_header_returns_422(self, client_with_key):
        resp = client_with_key.post("/retrain")
        assert resp.status_code == 422

    def test_wrong_key_returns_401(self, client_with_key):
        resp = client_with_key.post("/retrain", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_correct_key_accepted(self, client_with_key):
        with patch("src.recommender.api.build_training_data") as mock_train_data, \
             patch("src.recommender.api.train") as mock_train, \
             patch("src.recommender.api.save_model"):
            mock_df = MagicMock()
            mock_df.__len__ = lambda self: 100
            mock_df["brand"].nunique.return_value = 10
            mock_df["domain_type"].nunique.return_value = 5
            mock_train_data.return_value = mock_df
            mock_train.return_value = (MagicMock(), MagicMock(), ["f1", "f2"])

            resp = client_with_key.post("/retrain", headers={"X-API-Key": "test-secret"})
            assert resp.status_code == 200

    def test_server_misconfigured_returns_503(self, client_no_key):
        resp = client_no_key.post("/retrain", headers={"X-API-Key": "anything"})
        assert resp.status_code == 503
