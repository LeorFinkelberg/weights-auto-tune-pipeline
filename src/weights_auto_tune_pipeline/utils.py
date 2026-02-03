import typing as t
import json

import yt.wrapper as yt
from itertools import islice

from pathlib import Path
from weights_auto_tune_pipeline.constants import YtProxyClusterNames
from weights_auto_tune_pipeline.types_ import StrPath, StrTablePath


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


def read_data_from_yt_table(
    path_to_yt_table: StrTablePath,
    path_to_output: StrPath,
    format_: str = yt.JsonFormat(),
    proxy: t.Optional[str] = None,
    token: t.Optional[str] = None,
    start_row: t.Optional[int] = 0,
    end_row: t.Optional[int] = 1_000,
    config: t.Optional[dict[str, t.Any]] = None,
) -> None:
    yt_client: yt.YtClient = create_yt_client(
        proxy=proxy,
        token=token,
        config=config,
    )

    table = yt_client.read_table(
        table=path_to_yt_table,
        format=format_,
    )

    with Path.cwd().joinpath(path_to_output).open(mode="w", encoding="utf-8") as output:
        line: dict
        for line in islice(table, start_row, end_row):
            output.write(f"{json.dumps(line)}\n")


def write_data_to_yt():
    # TODO
    pass
