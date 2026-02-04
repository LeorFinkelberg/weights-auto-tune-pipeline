import typing as t
from pathlib import Path
import yt.wrapper as yt

from loguru import logger
from weights_auto_tune_pipeline.logging_ import setup_logging
from dataclasses import dataclass, field
from weights_auto_tune_pipeline.types_ import StrPath, StrTablePath
from weights_auto_tune_pipeline.utils import read_data_from_yt_table
from weights_auto_tune_pipeline.constants import Metrics

setup_logging()


@dataclass(frozen=True)
class PoolCache:
    path_to_yt_table: StrTablePath
    path_to_output: StrPath
    proxy: t.Optional[str] = None
    token: t.Optional[str] = None
    config: t.Optional[dict[str, t.Any]] = None
    start_row: t.Optional[int] = 0
    end_row: t.Optional[int] = 1_000
    format_: yt.JsonFormat = field(default_factory=yt.JsonFormat)

    def __post_init__(self) -> None:
        if not Path.cwd().joinpath(self.path_to_output).exists():
            read_data_from_yt_table(
                path_to_yt_table=self.path_to_yt_table,
                path_to_output=self.path_to_output,
                proxy=self.proxy,
                token=self.token,
                config=self.config,
                start_row=self.start_row,
                end_row=self.end_row,
                format_=self.format_,
            )
        else:
            logger.warning(
                f"File {str(self.path_to_output)!r} already exists. "
                "The file was not downloaded ..."
            )

    def compute_metric(self, metric: Metrics = Metrics.GAUC) -> float:
        match metric:
            case Metrics.GAUC:
                return 0.5
            case _:
                return -1.0

        gauc = 0.85
        logger.success(f"Done! GAUC: {gauc}")

        return gauc
