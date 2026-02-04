import abc
import typing as t
from dataclasses import dataclass
from auto_tune_weights_pipeline.columns import Columns
from auto_tune_weights_pipeline.target_config import TargetConfig

from auto_tune_weights_pipeline.types_ import StrTablePath


@dataclass(frozen=True)
class Metric(abc.ABC):
    path_to_pool_cache: StrTablePath

    @abc.abstractmethod
    def calculate_metric(
        self,
        target_configs: t.Union[list[TargetConfig], list[str], dict[str, TargetConfig]],
        session_col_name: str = Columns.RID_COL_NAME,
        score_col_name: str = Columns.SCORE_COL_NAME,
    ) -> float:
        """Computes metric."""

    def get_metric_name(self) -> str:
        """Gets metrics name."""
        return type(self).__name__.upper()
