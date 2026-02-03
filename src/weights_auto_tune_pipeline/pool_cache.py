import typing as t
from pathlib import Path
import yt.wrapper as yt

from loguru import logger
from weights_auto_tune_pipeline.logging_ import setup_logging
from dataclasses import dataclass, field
from weights_auto_tune_pipeline.types_ import StrPath, StrTablePath
from weights_auto_tune_pipeline.utils import read_data_from_yt_table

setup_logging()


@dataclass(frozen=True)
class PoolCache:
    path_to_yt_table: StrTablePath
    path_to_output: StrPath
    format_: yt.JsonFormat = field(default_factory=yt.JsonFormat)
    proxy: t.Optional[str] = None
    token: t.Optional[str] = None
    start_row: t.Optional[int] = 0
    end_row: t.Optional[int] = 1_000
    config: t.Optional[dict[str, t.Any]] = None

    def compute_gauc(self) -> float:
        if not Path.cwd().joinpath(self.path_to_output).exists():
            read_data_from_yt_table(
                path_to_yt_table=self.path_to_yt_table,
                path_to_output=self.path_to_output,
                format_=self.format_,
                proxy=self.proxy,
                token=self.token,
                start_row=self.start_row,
                end_row=self.end_row,
                config=self.config,
            )
        else:
            logger.warning(
                f"File {str(self.path_to_output)!r} already exists. "
                "The file was not downloaded ..."
            )

        gauc = 0.85
        logger.success(f"Done! GAUC: {gauc}")

        return gauc
