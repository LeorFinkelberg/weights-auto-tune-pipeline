import typing as t
import json

import yt.wrapper as yt
from itertools import islice

from pathlib import Path
from loguru import logger
from auto_tune_weights_pipeline.constants import YtProxyClusterNames
from auto_tune_weights_pipeline.types_ import StrPath, StrTablePath
from auto_tune_weights_pipeline.logging_ import setup_logging

setup_logging()


@logger.catch
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


@logger.catch
def read_data_from_yt_table(
    path_to_yt_table: StrTablePath,
    path_to_output: StrPath,
    proxy: t.Optional[str] = None,
    token: t.Optional[str] = None,
    config: t.Optional[dict[str, t.Any]] = None,
    start_row: t.Optional[int] = 0,
    end_row: t.Optional[int] = 1_000,
    format_: yt.JsonFormat = yt.JsonFormat(),
) -> None:
    """Reads table from YT Cypress."""

    yt_client: yt.YtClient = create_yt_client(
        proxy=proxy,
        token=token,
        config=config,
    )

    if not yt_client.exists(path_to_yt_table):
        logger.error(f"Table {path_to_yt_table!r} was not found ...")
        return

    table = yt_client.read_table(
        table=path_to_yt_table,
        format=format_,
    )

    _path_to_output = Path.cwd().joinpath(path_to_output)
    with _path_to_output.open(mode="w", encoding="utf-8") as output:
        line: dict
        for line in islice(table, start_row, end_row):
            output.write(f"{json.dumps(line)}\n")

        if _path_to_output.exists():
            logger.success(f"File {str(_path_to_output)!r} was successfully written")


@logger.catch
def write_data_to_yt():
    # TODO
    pass
