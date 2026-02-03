import pytest

from unittest.mock import Mock, patch, MagicMock
from weights_auto_tune_pipeline.utils import YtUtils
from weights_auto_tune_pipeline.constants import YtProxyClusterNames


class TestYtClientCreation:
    @pytest.fixture
    def mock_yt_client(self):
        with patch("yt.wrapper.YtClient") as mock_client:
            yield mock_client

    def test_yt_client_with_defaults(self, mock_yt_client):
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = YtUtils.create_yt_client()

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_JUPITER,
            token=None,
            config=None,
        )

        assert yt_client == mock_instance

    def test_yt_client_with_custom_proxy(self, mock_yt_client):
        mock_instance = MagicMock()
        mock_yt_client.return_value = mock_instance

        yt_client = YtUtils.create_yt_client(proxy=YtProxyClusterNames.YT_PROXY_SATURN)

        mock_yt_client.assert_called_once_with(
            proxy=YtProxyClusterNames.YT_PROXY_SATURN,
            token=None,
            config=None,
        )

        assert yt_client == mock_instance
