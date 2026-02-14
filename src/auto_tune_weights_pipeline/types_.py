import typing as t

from pathlib import Path
from yt.wrapper.ypath import FilePath, TablePath

from auto_tune_weights_pipeline.constants import Platforms


StrPath: t.TypeAlias = t.Union[str, Path]
StrFilePath: t.TypeAlias = t.Union[str, FilePath]
StrTablePath: t.TypeAlias = t.Union[str, TablePath]
TupleStrOrPlatforms: t.TypeAlias = tuple[t.Union[str, Platforms], ...]
