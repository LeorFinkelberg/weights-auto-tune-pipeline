import typing as t
import json

import numpy as np
import polars as pl

import yt.wrapper as yt

from pathlib import Path
from loguru import logger

from dataclasses import dataclass
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.constants import YtProxyClusterNames, SummaryLogFields
from auto_tune_weights_pipeline.types_ import StrPath, StrTablePath


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

        yt_client.run_map(
            _filter_mapper,
            source_table=self.path_to_yt_table,
            destination_table=self.path_to_temp_yt_table,
            format=yt.JsonFormat(),
            spec={"max_failed_job_count": 1},
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

                if _path_to_output.exists():
                    logger.success(
                        f"File {str(_path_to_output)!r} was successfully written"
                    )
        else:
            logger.warning(
                f"File {str(_path_to_output)!r} already exists. The file was not downloaded ..."
            )


class LogParser:
    def __init__(self, path_to_log_file: StrPath) -> None:
        self._auc_log = pl.read_ndjson(path_to_log_file)

    @staticmethod
    def convert_message_to_dict(message: str) -> dict:
        _message = (
            message.replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        return json.loads(_message)

    def get_value_from_summary_log_fild(
        self,
        target_name: str = "watch_coverage_30s",
        *,
        summary_log_fild: str = "group_details",
        field_name_with_records: str = "record",
        field_name_with_messages: str = "message",
        auc_message_pos: int = -10,
    ) -> t.Union[float, dict]:
        auc_message: str = self._auc_log.select(
            pl.col(field_name_with_records).struct.field(field_name_with_messages)
        ).to_dicts()[auc_message_pos][field_name_with_messages]
        auc_logs_parsed: dict = self.convert_message_to_dict(auc_message)
        _logs = auc_logs_parsed[target_name]

        if not summary_log_fild in SummaryLogFields.get_values():
            raise ValueError(f"Error! Not found field: {summary_log_fild!r}")

        return _logs[summary_log_fild]

    def read_logs_to_aucs(
        self,
        target_name: str = "watch_coverage_30s",
        *,
        field_name_with_records: str = "record",
        field_name_with_messages: str = "message",
        field_name_with_group_details: str = "group_details",
        auc_message_pos: int = -10,
    ) -> np.ndarray:
        """Reads app logs and converts to array of AUCs."""

        _logs = self.get_value_from_summary_log_fild(
            target_name,
            summary_log_fild=field_name_with_group_details,
            field_name_with_records=field_name_with_records,
            field_name_with_messages=field_name_with_messages,
            auc_message_pos=auc_message_pos,
        )

        return np.array([group["auc"] for group in _logs])
