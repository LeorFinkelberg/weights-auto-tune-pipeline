import typing as t

from pathlib import Path
from yt.wrapper.ypath import FilePath, TablePath

StrPath: t.TypeAlias = t.Union[str, Path]
StrFilePath: t.TypeAlias = t.Union[str, FilePath]
StrTablePath: t.TypeAlias = t.Union[str, TablePath]
