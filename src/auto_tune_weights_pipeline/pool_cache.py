import typing as t

import yt.wrapper as yt

from auto_tune_weights_pipeline.logging_ import setup_logging
from dataclasses import dataclass, field
from auto_tune_weights_pipeline.types_ import StrPath, StrTablePath
from auto_tune_weights_pipeline.utils import YtReader
from auto_tune_weights_pipeline.metrics.base import MetricNames
from auto_tune_weights_pipeline.metrics.gauc import GAUC
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.target_config import TargetConfig

setup_logging()


# TODO: remove
@dataclass(frozen=True)
class PoolCache:
    path_to_yt_table: StrTablePath
    path_to_temp_yt_table: StrTablePath
    path_to_output: StrPath
    proxy: t.Optional[str] = None
    token: t.Optional[str] = None
    config: t.Optional[dict[str, t.Any]] = None
    start_row: t.Optional[int] = 0
    end_row: t.Optional[int] = 1_000
    format_: yt.JsonFormat = field(default_factory=yt.JsonFormat)
    overwrite: bool = (True,)

    def __post_init__(self) -> None:
        yt_reader = YtReader(
            path_to_yt_table=self.path_to_yt_table,
            path_to_temp_yt_table=self.path_to_temp_yt_table,
            path_to_output=self.path_to_output,
        )
        yt_reader.read_big_yt_table()

    def calculate_metric(
        self,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
        nav_screen: str = "video_for_you",
        platform: str = "vk_video_android",
        formula_path: str = "fstorage:vk_video_266_1769078359_f",
        metric: MetricNames = MetricNames.GAUC,
        nav_screen_col_name: str = "navScreen",
        platform_col_name: str = "platform",
        formula_path_col_name: str = "formulaPath",
    ) -> dict[str, dict[str, t.Any]]:
        match metric:
            case MetricNames.GAUC:
                return GAUC(
                    path_to_pool_cache=self.path_to_output,
                ).calculate_metric(
                    target_configs=target_configs,
                    session_col_name=session_col_name,
                    score_col_name=score_col_name,
                    nav_screen=nav_screen,
                    platform=platform,
                    formula_path=formula_path,
                    nav_screen_col_name=nav_screen_col_name,
                    platform_col_name=platform_col_name,
                    formula_path_col_name=formula_path_col_name,
                )
            case MetricNames.QUERY_AUC_WEIGHTED:
                # TODO
                return {}
            case _:
                return {}
