import pytest

from unittest.mock import patch, MagicMock
from auto_tune_weights_pipeline.utils import create_yt_client
from auto_tune_weights_pipeline.constants import YtProxyClusterNames


class TestYtClientCreation:
    @pytest.fixture
    def mock_yt_client(self):
        with patch("yt.wrapper.YtClient") as mock_client:
            yield mock_client

    def test_yt_client_with_defaults(self, mock_yt_client):
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = create_yt_client()

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_JUPITER,
            token=None,
            config=None,
        )

        assert yt_client == mock_instance

    def test_yt_client_with_custom_proxy(self, mock_yt_client):
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = create_yt_client(proxy=YtProxyClusterNames.YT_PROXY_SATURN)

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_SATURN,
            token=None,
            config=None,
        )

        assert yt_client == mock_instance

    def test_create_client_with_token(self, mock_yt_client):
        test_token = "token"
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = create_yt_client(token=test_token)

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_JUPITER, token=test_token, config=None
        )

        assert yt_client == mock_instance

    def test_create_client_with_config(self, mock_yt_client):
        test_config = {
            "read_retries": 5,
            "write_retries": 3,
            "timeout": 30000,
            "enable_request_logging": True,
        }
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = create_yt_client(config=test_config)

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_JUPITER, token=None, config=test_config
        )

        assert yt_client == mock_instance

    def test_create_client_with_all_params(self, mock_yt_client):
        custom_proxy = YtProxyClusterNames.YT_PROXY_SATURN
        test_token = "secret-token"
        test_config = {"timeout": 60000}
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = create_yt_client(
            proxy=custom_proxy, token=test_token, config=test_config
        )

        mock_yt_client.assert_called_once_with(
            proxy=custom_proxy, token=test_token, config=test_config
        )

        assert yt_client == mock_instance
