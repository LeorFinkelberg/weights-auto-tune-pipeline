import typing as t
from dataclasses import dataclass

import yt.wrapper as yt

from pathlib import Path
from loguru import logger

from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.constants import YtProxyClusterNames
from auto_tune_weights_pipeline.types_ import StrPath, StrTablePath
from auto_tune_weights_pipeline.logging_ import setup_logging

setup_logging()


# TODO: fixme
@dataclass
class YtReader:
    path_to_yt_table: StrTablePath
    path_to_temp_yt_table: StrTablePath
    path_to_output: StrPath
    platform: str = "vk_video_android"
    nav_screen: str = "video_for_you"
    formula_path: str = "fstorage:vk_video_266_1769078359_f"
    proxy: t.Optional[str] = None
    token: t.Optional[str] = None
    config: t.Optional[dict[str, t.Any]] = None
    overwrite: bool = True

    @staticmethod
    def create_yt_client(
        proxy: t.Optional[str] = None,
        token: t.Optional[str] = None,
        config: t.Optional[dict[str, t.Any]] = None,
    ) -> yt.YtClient:
        """Creates YtClient."""

        return yt.YtClient(
            proxy=proxy or YtProxyClusterNames.YT_PROXY_JUPITER,
            token=token,
            config=config,
        )

    def read_big_yt_table(self) -> None:
        def _filter_mapper(row) -> t.Iterator[dict]:
            if (
                (row.get(Columns.PLATFORM_COL_NAME) == self.platform)
                and (row.get(Columns.NAV_SCREEN_COL_NAME) == self.nav_screen)
                and (row.get(Columns.FORMULA_PATH_COL_NAME) == self.formula_path)
            ):
                yield {
                    Columns.NAV_SCREEN_COL_NAME: row.get(Columns.NAV_SCREEN_COL_NAME),
                    Columns.RID_COL_NAME: row.get(Columns.RID_COL_NAME),
                    Columns.EVENTS_COL_NAME: row.get(Columns.EVENTS_COL_NAME),
                    Columns.VIEW_TIME_SEC_COL_NAME: row.get(
                        Columns.VIEW_TIME_SEC_COL_NAME
                    ),
                    Columns.PLATFORM_COL_NAME: row.get(Columns.PLATFORM_COL_NAME),
                    Columns.SCORE_COL_NAME: row.get(Columns.SCORE_COL_NAME),
                    Columns.FORMULA_PATH_COL_NAME: row.get(
                        Columns.FORMULA_PATH_COL_NAME
                    ),
                }

        logger.info("YtClient creating ...")
        yt_client: yt.YtClient = self.create_yt_client(
            proxy=self.proxy,
            token=self.token,
            config=self.config,
        )
        """# logger.error(yt_client.exists(self.path_to_yt_table).is_ok()) if
        not yt_client.exists(self.path_to_yt_table):

        logger.error(f"Table {self.path_to_yt_table!r} was not found ...")
        return
        """

        yt_client.run_map(
            _filter_mapper,
            source_table=self.path_to_yt_table,
            destination_table=self.path_to_temp_yt_table,
            format=yt.JsonFormat(),
            spec={
                # "mapper": {
                # "memory_limit": 4 * 1024 ** 3,
                # "cpu": 4,
                # },
                "max_failed_job_count": 1
            },
        )

        table = yt_client.read_table(self.path_to_temp_yt_table)

        _path_to_output = Path.cwd().joinpath(self.path_to_output)
        if (not _path_to_output.exists()) or (
            _path_to_output.exists() and self.overwrite
        ):
            with _path_to_output.open(mode="w", encoding="utf-8") as output:
                line: dict
                for line in table:
                    logger.debug(line)
                    # output.write(f"{json.dumps(line)}\n")

                if _path_to_output.exists():
                    logger.success(
                        f"File {str(_path_to_output)!r} was successfully written"
                    )
        else:
            logger.warning(
                f"File {str(_path_to_output)!r} already exists. The file was not downloaded ..."
            )
