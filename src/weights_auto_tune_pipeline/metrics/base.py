import abc
import typing as t
from dataclasses import dataclass
from enum import Enum, auto
from weights_auto_tune_pipeline.columns import Columns
from weights_auto_tune_pipeline.target_config import TargetConfig

from weights_auto_tune_pipeline.types_ import StrTablePath


class MetricNames(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()

    QUERY_AUC = auto()
    AUC_WEIGHTED = auto()
    QUERY_AUC_WEIGHTED = auto()
    AUC_WEIGHTED_NORMALIZED = auto()
    GAUC = auto()


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
