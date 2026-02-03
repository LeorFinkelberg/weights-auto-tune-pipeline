import typing as t
import yt.wrapper as yt

from weights_auto_tune_pipeline.constants import YtProxyClusterNames


class YtUtils:
    @staticmethod
    def read_data_from_yt():
        pass

    @staticmethod
    def create_yt_client(
        proxy: t.Optional[str] = None,
        token: t.Optional[str] = None,
        config: t.Optional[dict[str, t.Any]] = None,
    ) -> yt.YtClient:
        return yt.YtClient(
            proxy=proxy or YtProxyClusterNames.YT_PROXY_JUPITER,
            token=token,
            config=config,
        )

    @staticmethod
    def write_data_to_yt():
        pass


if __name__ == "__main__":
    pass
