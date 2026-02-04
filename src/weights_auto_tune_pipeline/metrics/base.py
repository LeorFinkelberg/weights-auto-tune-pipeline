import abc
from dataclasses import dataclass
from enum import Enum, auto

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
        nav_screen: str,
        platform: str,
        formula_path: str,
        metric: MetricNames,
        nav_screen_col_name: str,
        platform_col_name: str,
        formula_path_col_name: str,
    ) -> float:
        """Computes metric."""

    def get_metric_name(self) -> str:
        """Gets metrics name."""
        return type(self).__name__.upper()
