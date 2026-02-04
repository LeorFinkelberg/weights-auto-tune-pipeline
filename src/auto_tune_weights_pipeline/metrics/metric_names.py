from enum import Enum, auto


class MetricNames(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()

    QUERY_AUC = auto()
    AUC_WEIGHTED = auto()
    QUERY_AUC_WEIGHTED = auto()
    AUC_WEIGHTED_NORMALIZED = auto()
    GAUC = auto()
